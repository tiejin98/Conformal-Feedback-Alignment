"""Stage 3a: Weighted DPO training."""

import logging
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOConfig

from cfa.config import get_output_dir
from cfa.models.weighted_dpo import WeightedDPOTrainer

logger = logging.getLogger(__name__)


def run_train(config: dict):
    """Run weighted DPO training using uncertainty-weighted preference data.

    Args:
        config: Configuration dictionary.
    """
    model_cfg = config["model"]
    dpo_cfg = config["dpo"]
    hf_token = config.get("hf_token", None)
    fb_dir = get_output_dir(config, "feedback")

    data_path = str(fb_dir / "dpo_data_llama2_withuncertainty.json")
    logger.info(f"Loading DPO data from {data_path}")
    dataset = load_dataset("json", data_files=data_path, split="train")

    model_name = model_cfg.get("base_model_plain", model_cfg["base_model"])
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        cache_dir=model_cfg.get("cache_dir"),
        token=hf_token,
    )

    output_dir = model_cfg["dpo_output_dir"]
    dpo_config = DPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=dpo_cfg["per_device_train_batch_size"],
        learning_rate=dpo_cfg["learning_rate"],
        logging_steps=dpo_cfg["logging_steps"],
        save_steps=dpo_cfg["save_steps"],
        save_total_limit=dpo_cfg.get("save_total_limit", 2),
    )

    trainer = WeightedDPOTrainer(
        model=model,
        tokenizer=tokenizer,
        args=dpo_config,
        train_dataset=dataset,
    )

    logger.info(f"Starting weighted DPO training, output: {output_dir}")
    trainer.train()
    logger.info("DPO training complete")
