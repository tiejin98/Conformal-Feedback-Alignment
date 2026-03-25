"""Stage 1b: Sample multiple generations per prompt and score with GPT."""

import logging
import json
import numpy as np
import torch
from collections import defaultdict
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

from cfa.config import get_output_dir
from cfa.utils.scoring import get_openai_score
from cfa.utils.io import save_text, save_json, save_pickle

logger = logging.getLogger(__name__)


def run_generation(config: dict):
    """Generate multiple samples per prompt, score unique responses with GPT.

    For each calibration prompt, generates `sampling_num` responses and records
    frequency distributions. Unique responses are scored by GPT-4o. The same
    sampling is done for test prompts (without GPT scoring).

    Args:
        config: Configuration dictionary.
    """
    gen_cfg = config["generation"]
    model_cfg = config["model"]
    data_cfg = config["data"]
    output_dir = get_output_dir(config, "generation")

    device = gen_cfg.get("device", "cuda:0")
    calibration_size = gen_cfg["calibration_size"]
    sampling_num = gen_cfg["sampling_num"]
    num_return_sequences = gen_cfg.get("num_return_sequences", 1)

    np.random.seed(gen_cfg["seed"])
    torch.manual_seed(gen_cfg["seed"])

    # OpenAI client
    client = OpenAI(api_key=config["openai_api_key"])
    eval_model = config.get("evaluation", {}).get("model", "gpt-4o")

    # Load SFT model
    logger.info(f"Loading SFT model from {model_cfg['sft_checkpoint']}")
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["sft_checkpoint"], torch_dtype=torch.bfloat16
    )
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and split data
    dataset = load_dataset(data_cfg["dataset"], data_cfg["subset"])
    val_data = dataset["validation"]

    unique_data = {}
    for example in val_data:
        info = example["info"]
        summary = example["summary"]["text"]
        this_id = info["id"]
        if this_id not in unique_data:
            unique_data[this_id] = (info["post"], summary)

    questions, answers = [], []
    test_questions, test_answers = [], []
    max_words = data_cfg.get("max_post_words", 500)
    num = 0

    for the_id, (the_post, the_summary) in unique_data.items():
        words = the_post.split()
        if len(words) <= max_words:
            num += 1
            prompt_text = "Please summarize this text:\n" + the_post + "\nSummary:"
            if num <= calibration_size:
                questions.append(prompt_text)
                answers.append(the_summary)
            else:
                test_questions.append(prompt_text)
                test_answers.append(the_summary)

    logger.info(f"Calibration: {len(questions)}, Test: {len(test_questions)} samples")

    def generate_text(prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs["attention_mask"],
            do_sample=True,
            max_new_tokens=gen_cfg.get("max_new_tokens", 128),
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
            temperature=gen_cfg.get("temperature", 0.35),
        )
        transition_scores = model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )
        return outputs, transition_scores

    # --- Calibration set ---
    generated_responses = []
    accuracy_scores = []

    logger.info("Generating on Calibration Set...")
    for i in tqdm(range(calibration_size)):
        prompt_question = questions[i]
        reference_answer = answers[i]
        response_dict = defaultdict(int)

        for _ in range(sampling_num):
            response, score = generate_text(prompt_question)
            full_text = tokenizer.decode(response.sequences[0], skip_special_tokens=False)
            idx_found = full_text.find(prompt_question)
            if idx_found != -1:
                summary_text = full_text[idx_found + len(prompt_question):].strip()
            else:
                summary_text = full_text.strip()
            summary_text = summary_text.lower()
            response_dict[summary_text] += 1

        # Score unique summaries with GPT
        response_scores = {}
        for summary_text in response_dict.keys():
            openai_score = get_openai_score(
                client, prompt_question, summary_text, reference_answer, model=eval_model
            )
            response_scores[summary_text] = openai_score

        generated_responses.append(dict(response_dict))
        accuracy_scores.append(dict(response_scores))

    save_text(generated_responses, output_dir / "generation_llama2.txt")
    save_json(accuracy_scores, output_dir / "generation_llama2_accuracy.txt")
    logger.info("Calibration set generation done")

    # --- Test set ---
    response_dict_final = {}
    logger.info("Generating on Test Set...")
    with open(output_dir / "generation_test_llama2.txt", "w", encoding="utf-8") as file:
        for k in tqdm(range(len(test_questions))):
            response_dict_test = defaultdict(int)
            for _ in range(sampling_num):
                response, score = generate_text(test_questions[k])
                full_text = tokenizer.decode(response.sequences[0], skip_special_tokens=False)
                idx_found = full_text.find(test_questions[k])
                if idx_found != -1:
                    test_summary_text = full_text[idx_found + len(test_questions[k]):].strip()
                else:
                    test_summary_text = full_text.strip()
                test_summary_text = test_summary_text.lower()
                response_dict_test[test_summary_text] += 1

            response_dict_test = dict(response_dict_test)
            response_dict_final[test_questions[k]] = response_dict_test
            file.write(str(response_dict_test))
            file.write("\n")

    save_pickle(response_dict_final, output_dir / "response_dict_llama2.pkl")
    logger.info("Test set generation done")
