"""WeightedDPOTrainer: DPO with per-example uncertainty weights."""

import math
from contextlib import nullcontext
from typing import Any, Dict, Literal, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel

from trl import DPOTrainer, DPOConfig

try:
    from trl.trainer.dpo_types import FDivergenceType, FDivergenceConstants
    from trl.trainer.utils import cap_exp
except ImportError:
    class FDivergenceType:
        ALPHA_DIVERGENCE = "alpha"
        JS_DIVERGENCE = "js"

    class FDivergenceConstants:
        ALPHA_DIVERGENCE_COEF_DEFAULT = 1.0
        ALPHA_DIVERGENCE_COEF_KEY = "alpha_divergence_coef"

    def cap_exp(x):
        return torch.exp(torch.clamp(x, max=20))

try:
    from torch.cuda import amp
except ImportError:
    amp = None


class WeightedDPOTrainer(DPOTrainer):
    """DPOTrainer subclass that multiplies per-example loss by a 'weight' field.

    The dataset must have columns: ["prompt", "chosen", "rejected", "weight"].
    The weight field controls how much each preference pair contributes to the
    loss, enabling uncertainty-aware training where high-confidence pairs
    have higher weights.
    """

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
        num_items_in_batch=None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        compute_loss_context_manager = (
            amp.autocast("cuda") if (amp is not None and self._peft_has_been_casted_to_bf16) else nullcontext()
        )
        with compute_loss_context_manager:
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        loss = loss.to(self.args.device)
        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return loss, metrics
        return loss

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[list, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        metrics = {}
        model_output = self.concatenated_forward(model, batch)

        if "ref_chosen_logps" in batch and "ref_rejected_logps" in batch:
            ref_chosen_logps = batch["ref_chosen_logps"]
            ref_rejected_logps = batch["ref_rejected_logps"]
        else:
            ref_chosen_logps, ref_rejected_logps = self.compute_ref_log_probs(batch)

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            chosen_logps=model_output["chosen_logps"],
            rejected_logps=model_output["rejected_logps"],
            ref_chosen_logps=ref_chosen_logps,
            ref_rejected_logps=ref_rejected_logps,
            batch=batch,
        )

        loss = losses.mean()
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        metrics["reward_accuracy"] = reward_accuracies.mean().detach().cpu().item()

        return loss, metrics

    def dpo_loss(
        self,
        chosen_logps: torch.FloatTensor,
        rejected_logps: torch.FloatTensor,
        ref_chosen_logps: torch.FloatTensor,
        ref_rejected_logps: torch.FloatTensor,
        batch: Dict[str, torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute DPO loss with optional per-example weighting.

        Same as standard DPO loss, but multiplies final losses by batch['weight']
        when present. This is the core mechanism for uncertainty-aware training.
        """
        device = self.accelerator.device

        chosen_logratios = chosen_logps.to(device) - (not self.reference_free) * ref_chosen_logps.to(device)
        rejected_logratios = rejected_logps.to(device) - (not self.reference_free) * ref_rejected_logps.to(device)

        if self.f_divergence_type == FDivergenceType.ALPHA_DIVERGENCE:
            alpha_coef = FDivergenceConstants.ALPHA_DIVERGENCE_COEF_DEFAULT
            if self.f_divergence_params and FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY in self.f_divergence_params:
                alpha_coef = float(self.f_divergence_params[FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY])
            logits = (cap_exp(rejected_logratios * -alpha_coef) - cap_exp(chosen_logratios * -alpha_coef)) / alpha_coef
        else:
            logratios = chosen_logps - rejected_logps
            if self.reference_free:
                ref_logratios = torch.tensor([0], dtype=logratios.dtype, device=logratios.device)
            else:
                ref_logratios = ref_chosen_logps - ref_rejected_logps
            logratios = logratios.to(device)
            ref_logratios = ref_logratios.to(device)
            logits = logratios - ref_logratios

            if self.f_divergence_type == FDivergenceType.JS_DIVERGENCE:
                logits -= F.softplus(chosen_logratios) - F.softplus(rejected_logratios)

        # Compute losses based on configured loss type
        if self.loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "robust":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                + F.logsigmoid(-self.beta * logits) * self.label_smoothing
            ) / (1 - 2 * self.label_smoothing)
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        elif self.loss_type == "ipo":
            losses = (logits - 1 / (2 * self.beta)) ** 2
        elif self.loss_type == "exo_pair":
            if self.label_smoothing == 0:
                self.label_smoothing = 1e-3
            losses = (self.beta * logits).sigmoid() * (
                F.logsigmoid(self.beta * logits) - math.log(1 - self.label_smoothing)
            ) + (-self.beta * logits).sigmoid() * (
                F.logsigmoid(-self.beta * logits) - math.log(self.label_smoothing)
            )
        elif self.loss_type == "bco_pair":
            chosen_logratios = chosen_logps - ref_chosen_logps
            rejected_logratios = rejected_logps - ref_rejected_logps
            chosen_rewards = self.beta * chosen_logratios
            rejected_rewards = self.beta * rejected_logratios
            rewards = torch.cat((chosen_rewards, rejected_rewards), 0).mean().detach()
            self.running.update(rewards)
            delta = self.running.mean
            losses = -F.logsigmoid((self.beta * chosen_logratios) - delta) - F.logsigmoid(
                -(self.beta * rejected_logratios - delta)
            )
        elif self.loss_type == "sppo_hard":
            a = chosen_logps - ref_chosen_logps
            b = rejected_logps - ref_rejected_logps
            losses = (a - 0.5 / self.beta) ** 2 + (b + 0.5 / self.beta) ** 2
        elif self.loss_type == "nca_pair":
            chosen_rewards = (chosen_logps - ref_chosen_logps) * self.beta
            rejected_rewards = (rejected_logps - ref_rejected_logps) * self.beta
            losses = (
                -F.logsigmoid(chosen_rewards)
                - 0.5 * F.logsigmoid(-chosen_rewards)
                - 0.5 * F.logsigmoid(-rejected_rewards)
            )
        elif self.loss_type == "aot_pair":
            chosen_logratios = chosen_logps - ref_chosen_logps
            rejected_logratios = rejected_logps - ref_rejected_logps
            chosen_logratios_sorted, _ = torch.sort(chosen_logratios, dim=0)
            rejected_logratios_sorted, _ = torch.sort(rejected_logratios, dim=0)
            delta = chosen_logratios_sorted - rejected_logratios_sorted
            losses = (
                -F.logsigmoid(self.beta * delta) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * delta) * self.label_smoothing
            )
        elif self.loss_type == "aot":
            logratios = chosen_logps - rejected_logps
            ref_logratios = ref_chosen_logps - ref_rejected_logps
            logratios_sorted, _ = torch.sort(logratios, dim=0)
            ref_logratios_sorted, _ = torch.sort(ref_logratios, dim=0)
            delta = logratios_sorted - ref_logratios_sorted
            losses = (
                -F.logsigmoid(self.beta * delta) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * delta) * self.label_smoothing
            )
        elif self.loss_type == "apo_zero":
            losses_chosen = 1 - F.sigmoid(self.beta * chosen_logratios)
            losses_rejected = F.sigmoid(self.beta * rejected_logratios)
            losses = losses_chosen + losses_rejected
        elif self.loss_type == "apo_down":
            losses_chosen = F.sigmoid(self.beta * chosen_logratios)
            losses_rejected = 1 - F.sigmoid(self.beta * (chosen_logratios - rejected_logratios))
            losses = losses_chosen + losses_rejected
        elif self.loss_type == "discopop":
            logratios = chosen_logps - rejected_logps
            ref_logratios = ref_chosen_logps - ref_rejected_logps
            logits = (logratios - ref_logratios) * self.beta
            log_ratio_modulation = torch.sigmoid(logits / self.args.discopop_tau)
            logistic_component = -F.logsigmoid(logits)
            exp_component = torch.exp(-logits)
            losses = logistic_component * (1 - log_ratio_modulation) + exp_component * log_ratio_modulation
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ["
                "'sigmoid', 'hinge', 'ipo', 'exo_pair', 'nca_pair', 'robust', "
                "'bco_pair', 'sppo_hard', 'aot', 'aot_pair', 'discopop', "
                "'apo_zero', 'apo_down']"
            )

        # Apply per-example uncertainty weights
        if batch is not None and "weight" in batch:
            w = batch["weight"].to(losses.device)
            losses = losses * w

        chosen_rewards = self.beta * (chosen_logps.to(device) - ref_chosen_logps.to(device)).detach()
        rejected_rewards = self.beta * (rejected_logps.to(device) - ref_rejected_logps.to(device)).detach()

        return losses, chosen_rewards, rejected_rewards
