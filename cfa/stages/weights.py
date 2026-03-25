"""Stage 2b: Assign uncertainty weights to DPO pairs using CP prediction sets."""

import json
import logging
from cfa.config import get_output_dir
from cfa.utils.io import load_json, load_jsonl, save_jsonl

logger = logging.getLogger(__name__)


def run_assign_weights(config: dict):
    """Assign uncertainty-based weights to each DPO training pair.

    Responses in the 50% coverage prediction set get weight 0.5.
    Responses only in the 80% coverage set get weight 0.8.
    Responses in both get weight (0.5+0.8)/2 = 0.65.
    Each DPO pair's weight = average of chosen and rejected weights.

    Args:
        config: Configuration dictionary.
    """
    cal_dir = get_output_dir(config, "calibration")
    fb_dir = get_output_dir(config, "feedback")
    threshold = config["conformal"].get("accuracy_threshold", 0.7)

    # Load prediction sets
    five_path = cal_dir / f"prediction_set_quantile0.5_threshold{threshold}_llama2.json"
    eight_path = cal_dir / f"prediction_set_quantile0.2_threshold{threshold}_llama2.json"

    five_coverage = load_json(five_path)
    eight_coverage = load_json(eight_path)

    # Build weight dictionary
    weight_dict = {}

    for gen_list in five_coverage:
        for sample in gen_list:
            weight_dict[sample] = 0.5

    for gen_list in eight_coverage:
        for sample in gen_list:
            if sample not in weight_dict:
                weight_dict[sample] = 0.8
            else:
                weight_dict[sample] = (0.5 + 0.8) / 2

    logger.info(f"Weight dict: {len(weight_dict)} unique responses")

    # Load DPO data and assign weights
    dpo_data = load_jsonl(fb_dir / "dpo_data_llama2.json")
    weighted_data = []
    for data in dpo_data:
        weight = (weight_dict.get(data["chosen"], 0) + weight_dict.get(data["rejected"], 0)) / 2
        data["weight"] = weight
        weighted_data.append(data)

    save_jsonl(weighted_data, fb_dir / "dpo_data_llama2_withuncertainty.json")
    logger.info(f"Saved {len(weighted_data)} weighted DPO pairs")
