import math
from contextlib import nullcontext
from typing import Any, Dict, Literal, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.trainer_pt_utils import LabelSmoother
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

# TRL imports:
# Make sure you have installed: pip install trl
from trl import DPOTrainer, AutoModelForCausalLMWithValueHead, DPOConfig

# Optional TRL items for advanced f-divergences, remove if not needed
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

# Amp for mixed-precision
try:
    from torch.cuda import amp
except ImportError:
    amp = None


class WeightedDPOTrainer(DPOTrainer):
    """
    A subclass of TRL's DPOTrainer that multiplies the per-example DPO loss
    by a 'weight' field in each batch item.

    Usage:
      1) Make sure your dataset has columns:
            ["prompt", "chosen", "rejected", "weight"]
      2) Use WeightedDPOTrainer exactly like you would DPOTrainer.
    """

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
        num_items_in_batch=None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Same as the original DPOTrainer, except it calls our overridden
        get_batch_loss_metrics, which then calls our custom dpo_loss.
        """
        # If using PEFT + BF16, we need an autocast context. Otherwise, no-op.
        compute_loss_context_manager = (
            amp.autocast("cuda") if (amp is not None and self._peft_has_been_casted_to_bf16) else nullcontext()
        )
        with compute_loss_context_manager:
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        # Move loss to the device the trainer expects
        loss = loss.to(self.args.device)

        # Force log metrics
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
        """
        Computes the DPO loss and relevant metrics for the batch.
        Overridden to pass `batch` forward to dpo_loss so we can apply weights.
        """
        metrics = {}

        # The standard forward pass that computes chosen_logps, rejected_logps, etc.
        model_output = self.concatenated_forward(model, batch)

        # If 'ref_chosen_logps' and 'ref_rejected_logps' exist in batch, use them.
        # Otherwise, compute them from a reference model.
        if "ref_chosen_logps" in batch and "ref_rejected_logps" in batch:
            ref_chosen_logps = batch["ref_chosen_logps"]
            ref_rejected_logps = batch["ref_rejected_logps"]
        else:
            ref_chosen_logps, ref_rejected_logps = self.compute_ref_log_probs(batch)

        # Use our custom dpo_loss that applies per-example weighting
        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            chosen_logps=model_output["chosen_logps"],
            rejected_logps=model_output["rejected_logps"],
            ref_chosen_logps=ref_chosen_logps,
            ref_rejected_logps=ref_rejected_logps,
            batch=batch,  # <---- pass batch to dpo_loss
        )

        # We typically average the per-example losses here
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
        """
        Same logic as TRL's DPOTrainer.dpo_loss, but at the end we multiply
        'losses' by batch['weight'] if present.
        """
        device = self.accelerator.device

        # Calculate log ratios for chosen and rejected
        chosen_logratios = chosen_logps.to(device) - (not self.reference_free) * ref_chosen_logps.to(device)
        rejected_logratios = rejected_logps.to(device) - (not self.reference_free) * ref_rejected_logps.to(device)

        # We handle fancy f-divergence types, or fall back to standard DPO
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

        # Now compute the final losses for each example
        # (depending on the configured loss_type)
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
            # eqn (17) from the IPO paper, param self.beta is the regularizer
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
            # Hard version of sppo (Equation 4.7 from the paper)
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
            # eqn (7) from the APO paper
            losses_chosen = 1 - F.sigmoid(self.beta * chosen_logratios)
            losses_rejected = F.sigmoid(self.beta * rejected_logratios)
            losses = losses_chosen + losses_rejected
        elif self.loss_type == "apo_down":
            # eqn (8) from the APO paper
            losses_chosen = F.sigmoid(self.beta * chosen_logratios)
            losses_rejected = 1 - F.sigmoid(self.beta * (chosen_logratios - rejected_logratios))
            losses = losses_chosen + losses_rejected
        elif self.loss_type == "discopop":
            # eqn (5) from the DiscoPOP paper
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

        # Weighted or not?
        if batch is not None and "weight" in batch:
            w = batch["weight"].to(losses.device)
            # Multiply the per-example losses by weight
            losses = losses * w

        # For the returns, we also compute chosen_rewards and rejected_rewards
        chosen_rewards = self.beta * (chosen_logps.to(device) - ref_chosen_logps.to(device)).detach()
        rejected_rewards = self.beta * (rejected_logps.to(device) - ref_rejected_logps.to(device)).detach()

        return losses, chosen_rewards, rejected_rewards
