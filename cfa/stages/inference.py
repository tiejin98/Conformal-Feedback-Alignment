"""Stage 3b: Generate answers on the test set using the trained RLUF model."""

import logging
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from cfa.config import get_output_dir
from cfa.utils.io import save_pickle

logger = logging.getLogger(__name__)


def run_inference(config: dict):
    """Generate summaries on the test set using the trained RLUF model.

    Args:
        config: Configuration dictionary.
    """
    model_cfg = config["model"]
    gen_cfg = config["generation"]
    data_cfg = config["data"]
    hf_token = config.get("hf_token", None)
    output_dir = get_output_dir(config, "inference")
    device = gen_cfg.get("device", "cuda:0")

    np.random.seed(gen_cfg["seed"])
    torch.manual_seed(gen_cfg["seed"])

    logger.info(f"Loading RLUF model from {model_cfg['dpo_output_dir']}")
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["dpo_output_dir"], torch_dtype=torch.bfloat16
    )
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model"], token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load test data
    dataset = load_dataset(data_cfg["dataset"], data_cfg["subset"])
    val_data = dataset["test"]

    unique_data = {}
    for example in val_data:
        info = example["info"]
        this_id = info["id"]
        if this_id not in unique_data:
            unique_data[this_id] = info["article"]

    # Build prompts
    for key in unique_data.keys():
        unique_data[key] = "Please summarize this text:\n" + unique_data[key] + "\nSummary:"

    logger.info(f"Generating for {len(unique_data)} test samples...")

    response_res = {}
    for key in tqdm(unique_data.keys(), desc="Inference"):
        prompt = unique_data[key]
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_length = inputs.input_ids.shape[1]
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs["attention_mask"],
            max_new_tokens=200,
        )[0]
        generated_text = tokenizer.decode(outputs[input_length:], skip_special_tokens=True)
        response_res[key] = generated_text

    save_pickle(unique_data, output_dir / "test_dict_question.pkl")
    save_pickle(response_res, output_dir / "test_dict_RLUF.pkl")
    logger.info(f"Inference complete. Saved {len(response_res)} responses")
