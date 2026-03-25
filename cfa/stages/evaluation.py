"""Stage 3c: Evaluate generated responses using GPT-4o."""

import re
import time
import logging
from tqdm import tqdm
from openai import OpenAI

from cfa.config import get_output_dir
from cfa.utils.io import load_pickle, save_pickle

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert evaluator. You are given an original input (question or source text) "
    "and an AI-generated response (answer or summary). Your task is to evaluate the response "
    "based on four criteria: Accuracy, Relevance, Completeness, and Expression. Each criterion "
    "should be scored from 0 to 10 in increments of 0.5. Provide a brief justification for each "
    "score. Then, calculate the average of the four scores and present it as the Overall Score.\n\n"
    "Scoring Criteria:\n"
    "Accuracy (Acc): Does the response accurately reflect the content and intent of the original prompt?\n"
    "Relevance (Rel): Is the response closely aligned with the topic and requirements of the prompt?\n"
    "Completeness (Comp): Does the response address all essential aspects or key points in the prompt?\n"
    "Expression (Expr): Is the response clear, well-written, and easy to understand?\n\n"
    "Please ONLY return the four line-scores and the Overall Score, in this exact format:\n\n"
    "**Accuracy (Acc):** [score]/10  \n"
    "**Relevance (Rel):** [score]/10  \n"
    "**Completeness (Comp):** [score]/10  \n"
    "**Expression (Expr):** [score]/10  \n"
    "**Overall Score:** [average]/10\n"
)


def run_evaluation(config: dict):
    """Evaluate RLUF model outputs using GPT-4o on 4 criteria.

    Scores: Accuracy, Relevance, Completeness, Expression (each 0-10).

    Args:
        config: Configuration dictionary.
    """
    eval_cfg = config.get("evaluation", {})
    inf_dir = get_output_dir(config, "inference")
    eval_dir = get_output_dir(config, "evaluation")

    model_name = eval_cfg.get("model", "gpt-4o")
    rate_limit_pause = eval_cfg.get("rate_limit_pause", 0.3)

    client = OpenAI(api_key=config["openai_api_key"])

    questions = load_pickle(inf_dir / "test_dict_question.pkl")
    answers = load_pickle(inf_dir / "test_dict_RLUF.pkl")

    scores = {}

    def request_scores_once(prompt: str):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            return resp.choices[0].message.content
        except Exception as e:
            logger.warning(f"OpenAI API error: {e}")
            return None

    logger.info(f"Evaluating {len(questions)} samples with {model_name}...")

    for key in tqdm(list(questions.keys()), desc="Evaluation"):
        instruction = questions[key]
        ai_response = answers.get(key, "")
        user_prompt = (
            f"Original Input:\n{instruction}\n\n"
            f"AI Response:\n{ai_response}\n\n"
            "Please evaluate the response as described above."
        )

        parsed = False
        for attempt in range(2):
            text = request_scores_once(user_prompt)
            if text is None:
                break

            m = re.search(
                r"\*\*Accuracy \(Acc\):\*\*\s*([\d.]+)/10.*?"
                r"\*\*Relevance \(Rel\):\*\*\s*([\d.]+)/10.*?"
                r"\*\*Completeness \(Comp\):\*\*\s*([\d.]+)/10.*?"
                r"\*\*Expression \(Expr\):\*\*\s*([\d.]+)/10.*?"
                r"\*\*Overall Score:\*\*\s*([\d.]+)/10",
                text, flags=re.S | re.I,
            )

            if m:
                acc, rel, comp, expr, overall = map(float, m.groups())
                scores[key] = {
                    "Acc": acc, "Rel": rel, "Comp": comp,
                    "Expr": expr, "Overall": overall,
                }
                parsed = True
                break

            if attempt == 0:
                logger.warning(f"Could not parse scores for key {key!r}; retrying...")
                time.sleep(rate_limit_pause)

        if not parsed:
            logger.error(f"Failed to parse scores for key {key!r}")
            scores[key] = None

        time.sleep(rate_limit_pause)

    save_pickle(scores, eval_dir / "evaluation_scores_llama2.pkl")
    logger.info(f"Evaluation complete. Scored {len(scores)} items")

    # Print summary
    valid = [s for s in scores.values() if s is not None]
    if valid:
        avg_overall = sum(s["Overall"] for s in valid) / len(valid)
        logger.info(f"Average Overall Score: {avg_overall:.2f}/10 ({len(valid)} valid)")
