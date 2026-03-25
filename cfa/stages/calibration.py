"""Stage 1c: Conformal Prediction calibration and inference."""

import logging
import math
import random
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from gensim.test.utils import common_texts
from gensim.models import FastText

from cfa.config import get_output_dir
from cfa.utils.text_processing import process_list_of_dicts
from cfa.utils.scoring import compute_cp_score
from cfa.utils.io import load_text_as_literal, load_text_lines_as_literals, save_json

logger = logging.getLogger(__name__)


def run_calibration(config: dict):
    """Run conformal prediction calibration and apply to test set.

    Grid-searches over (weight, weight_2) hyperparameters, calibrates the
    quantile threshold on validation data, and produces prediction sets
    for the test set.

    Args:
        config: Configuration dictionary.
    """
    cp_cfg = config["conformal"]
    gen_dir = get_output_dir(config, "generation")
    cal_dir = get_output_dir(config, "calibration")

    quantile_bars = cp_cfg.get("quantile_bars", [0.2, 0.5])
    accuracy_threshold = cp_cfg.get("accuracy_threshold", 0.7)
    weights_range = cp_cfg.get("weights_range", [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    random_count = cp_cfg.get("random_count", 50)
    split_num = cp_cfg.get("split_num", 3)
    sim_vector_size = cp_cfg.get("similarity_vector_size", 200)

    # Initialize similarity model
    logger.info("Initializing FastText similarity model...")
    similarity_model = FastText(sentences=common_texts, vector_size=sim_vector_size, min_count=1)

    for quantile_bar in quantile_bars:
        logger.info(f"=== Running calibration for quantile_bar={quantile_bar} ===")
        _run_single_quantile(
            gen_dir, cal_dir, quantile_bar, accuracy_threshold,
            weights_range, random_count, split_num, similarity_model,
        )


def _run_single_quantile(gen_dir, cal_dir, quantile_bar, accuracy_threshold,
                          weights_range, random_count, split_num, similarity_model):
    """Run calibration for a single quantile_bar value."""
    # Load calibration data
    correct_answers_raw = load_text_as_literal(gen_dir / "generation_llama2.txt")
    generation_raw = load_text_as_literal(gen_dir / "generation_llama2_accuracy.txt")

    correct_answers = process_list_of_dicts(correct_answers_raw)
    generation = process_list_of_dicts(generation_raw)

    best_av_size = float("inf")
    best_params = [0, 0]
    best_quantile_value = 0
    results = []

    for weight in weights_range:
        for weight_2 in weights_range:
            combined_data = list(zip(correct_answers, generation))

            total_correct_sum = 0
            total_size_sum = 0
            total_val_size_sum = 0
            total_question = 0
            total_val_question = 0

            for _ in range(random_count):
                random.shuffle(combined_data)
                correct_shuffled, gen_freq = zip(*combined_data)
                nonconformity_scores = []
                test_set, test_correct = [], []
                val_set, val_correct = [], []
                bar = 0

                for index, dict_of_freq in enumerate(gen_freq):
                    if index % split_num == 0:
                        # Calibration split
                        correct_dict = correct_shuffled[index]
                        context, accuracy = next(iter(correct_dict.items()))
                        is_correct = accuracy >= accuracy_threshold

                        if is_correct and context in dict_of_freq:
                            score_dict, _ = compute_cp_score(
                                dict_of_freq, weight, weight_2, similarity_model
                            )
                            nonconformity_scores.append(score_dict[context])
                        else:
                            nonconformity_scores.append(20)
                        bar += 1
                    elif index % split_num == 1:
                        val_set.append(gen_freq[index])
                        val_correct.append(correct_shuffled[index])
                    else:
                        test_set.append(gen_freq[index])
                        test_correct.append(correct_shuffled[index])

                if bar == 0:
                    continue

                quantile = np.ceil((bar + 1) * (1 - quantile_bar)) / bar * 100
                sorted_scores = sorted(nonconformity_scores, reverse=True)
                quantile_value = np.percentile(sorted_scores, quantile)

                # Validation set size
                for dict_of_freq in val_set:
                    result, _ = compute_cp_score(dict_of_freq, weight, weight_2, similarity_model)
                    pred = [k for k, s in result.items() if s <= quantile_value]
                    total_val_size_sum += len(pred)
                    total_val_question += 1

                # Test set accuracy
                for k, dict_of_freq in enumerate(test_set):
                    total_question += 1
                    result, _ = compute_cp_score(dict_of_freq, weight, weight_2, similarity_model)
                    pred = [key for key, s in result.items() if s <= quantile_value]
                    total_size_sum += len(pred)

                    correct_dict = test_correct[k]
                    context, accuracy = next(iter(correct_dict.items()))
                    if accuracy >= accuracy_threshold and context in pred:
                        total_correct_sum += 1

            if total_val_question > 0 and total_question > 0:
                avg_val_size = total_val_size_sum / random_count / total_val_question
                avg_test_size = total_size_sum / random_count / total_question
                avg_accuracy = total_correct_sum / total_question / random_count

                logger.info(
                    f"w={weight:.1f}, w2={weight_2:.1f} | "
                    f"val_size={avg_val_size:.2f}, test_size={avg_test_size:.2f}, "
                    f"accuracy={avg_accuracy:.3f}"
                )

                if avg_val_size < best_av_size:
                    best_av_size = avg_val_size
                    best_params = [weight, weight_2]
                    best_quantile_value = quantile_value

                results.append([avg_test_size, avg_accuracy])

    logger.info(f"Best params: weight={best_params[0]}, weight_2={best_params[1]}")
    logger.info(f"Best avg val size: {best_av_size:.2f}")
    logger.info(f"Best quantile value: {best_quantile_value:.4f}")

    # Apply to test set
    logger.info("Applying conformal prediction to test set...")
    test_generation = load_text_lines_as_literals(gen_dir / "generation_test_llama2.txt")

    prediction_set = []
    for dict_of_freq in tqdm(test_generation, desc="CP inference"):
        result, _ = compute_cp_score(dict_of_freq, best_params[0], best_params[1], similarity_model)
        pred = [key for key, score in result.items() if score <= best_quantile_value]
        prediction_set.append(pred)

    output_path = cal_dir / f"prediction_set_quantile{quantile_bar}_threshold{accuracy_threshold}_llama2.json"
    save_json(prediction_set, output_path)
    logger.info(f"Prediction set saved: {output_path}")
