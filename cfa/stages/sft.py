"""Stage 1a: Supervised Fine-Tuning on summarization data."""

import logging
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

from cfa.config import get_output_dir

logger = logging.getLogger(__name__)


def run_sft(config: dict):
    """Run SFT training on the summarization dataset.

    Loads the openai/summarize_from_feedback validation set, constructs
    prompt-summary pairs, and fine-tunes the base LLM with loss computed
    only on the summary tokens.

    Args:
        config: Configuration dictionary.
    """
    hf_token = config.get("hf_token", None)
    model_cfg = config["model"]
    sft_cfg = config["sft"]
    data_cfg = config["data"]

    logger.info("Loading dataset...")
    dataset = load_dataset(data_cfg["dataset"], data_cfg["subset"])
    val_data = dataset["validation"]

    # Deduplicate by post ID
    unique_data = {}
    for example in val_data:
        info = example["info"]
        summary = example["summary"]["text"]
        this_id = info["id"]
        if this_id not in unique_data:
            unique_data[this_id] = (info["post"], summary)

    # Build training data
    data_list = []
    for _id, (post, summary) in unique_data.items():
        prompt = "Please summarize this text:\n" + post + "\nSummary:"
        full_text = prompt + " " + summary
        data_list.append({"prompt": prompt, "target": summary, "text": full_text})

    train_dataset = Dataset.from_list(data_list)
    logger.info(f"Training on {len(train_dataset)} unique samples")

    # Load model and tokenizer
    model_name = model_cfg["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        cache_dir=model_cfg.get("cache_dir"),
        token=hf_token,
    )

    max_length = sft_cfg.get("max_length", 1024)

    def tokenize_and_mask(example):
        full = tokenizer(example["text"], truncation=True, max_length=max_length)
        prompt_tokens = tokenizer(example["prompt"], truncation=True, max_length=max_length)["input_ids"]
        full_ids = full["input_ids"]
        labels = full_ids.copy()
        # Mask prompt tokens so loss is only on summary
        for i in range(len(prompt_tokens)):
            labels[i] = -100
        full["labels"] = labels
        return full

    tokenized_dataset = train_dataset.map(
        tokenize_and_mask, batched=False, remove_columns=["prompt", "target", "text"]
    )

    def custom_data_collator(features):
        batch = tokenizer.pad(
            {"input_ids": [f["input_ids"] for f in features],
             "attention_mask": [f["attention_mask"] for f in features]},
            return_tensors="pt",
        )
        max_length = batch["input_ids"].shape[1]
        padded_labels = []
        for f in features:
            labels = f["labels"]
            pad_length = max_length - len(labels)
            padded_labels.append(labels + [-100] * pad_length)
        batch["labels"] = torch.tensor(padded_labels)
        return batch

    output_dir = model_cfg["sft_output_dir"]
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=sft_cfg["num_train_epochs"],
        per_device_train_batch_size=sft_cfg["per_device_train_batch_size"],
        learning_rate=sft_cfg["learning_rate"],
        save_steps=sft_cfg["save_steps"],
        save_total_limit=sft_cfg["save_total_limit"],
        logging_steps=sft_cfg["logging_steps"],
        bf16=sft_cfg.get("bf16", True),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=custom_data_collator,
    )

    logger.info(f"Starting SFT training, output: {output_dir}")
    trainer.train()
    logger.info("SFT training complete")
