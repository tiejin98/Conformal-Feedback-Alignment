"""Stage 2a: AI pairwise preference annotation using AlpacaFarm."""

import logging
import itertools
import openai
from cfa.config import get_output_dir
from cfa.utils.io import load_pickle, save_jsonl

logger = logging.getLogger(__name__)


def run_feedback(config: dict):
    """Generate pairwise preference annotations for DPO training.

    Loads test set responses, creates all pairwise combinations per prompt,
    and uses AlpacaFarm's PairwiseAutoAnnotator (GPT-4 backend) to determine
    preferences. Outputs DPO-format data with prompt/chosen/rejected.

    Args:
        config: Configuration dictionary.
    """
    gen_dir = get_output_dir(config, "generation")
    fb_dir = get_output_dir(config, "feedback")

    openai.api_key = config["openai_api_key"]

    # Load test set responses
    response_dict = load_pickle(gen_dir / "response_dict_llama2.pkl")

    # Build pairwise comparison format
    adjust_format = []
    for key in response_dict.keys():
        second_dict = response_dict[key]
        response_list = list(second_dict.keys())
        if len(response_list) < 2:
            continue
        all_pairs = list(itertools.combinations(response_list, 2))
        for pair in all_pairs:
            adjust_format.append({
                "instruction": key,
                "input": "",
                "output_1": pair[0],
                "output_2": pair[1],
            })

    logger.info(f"Created {len(adjust_format)} pairwise comparisons")

    # Run AlpacaFarm annotation
    from alpaca_farm.auto_annotations import PairwiseAutoAnnotator
    annotator = PairwiseAutoAnnotator()
    annotated = annotator.annotate_pairs(adjust_format)

    # Convert to DPO format
    new_data = []
    for entry in annotated:
        prompt = entry["instruction"]
        if entry["preference"] == 1:
            chosen = entry["output_1"]
            rejected = entry["output_2"]
        else:
            chosen = entry["output_2"]
            rejected = entry["output_1"]

        new_data.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        })

    save_jsonl(new_data, fb_dir / "dpo_data_llama2.json")
    logger.info(f"Saved {len(new_data)} DPO pairs")
