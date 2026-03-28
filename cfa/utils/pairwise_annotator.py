"""Pairwise preference annotation using OpenAI API.

Replaces alpaca-farm's PairwiseAutoAnnotator with a direct OpenAI call,
using the same prompt template and debiasing strategy.
"""

import ast
import hashlib
import logging
import random
import time
from typing import Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "You are a helpful assistant, that ranks models by the quality of their answers."

USER_PROMPT_TEMPLATE = """I want you to create a leaderboard of different of large-language models. To do so, I will give you the instructions (prompts) given to the models, and the responses of two models. Please rank the models based on which responses would be preferred by humans. All inputs and outputs should be python dictionaries.

Here is the prompt:
{{"instruction": \"""{instruction}\"\"\"}}

Here are the outputs of the models:
[{{"model": "model_1", "answer": \"""{output_1}\"\"\"}}, {{"model": "model_2", "answer": \"""{output_2}\"\"\"}}]

Now please rank the models by the quality of their answers, so that the model with rank 1 has the best output. Then return a list of the model names and ranks, i.e., produce the following output:
[{{'model': <model-name>, 'rank': <model-rank>}}, {{'model': <model-name>, 'rank': <model-rank>}}]

Your response must be a valid Python dictionary and should contain nothing else because we will directly execute it in Python. Please provide the ranking that the majority of humans would give."""


def _should_swap(instruction: str) -> bool:
    """Deterministically decide whether to swap outputs based on instruction hash."""
    seed = int(hashlib.md5(instruction.encode()).hexdigest(), 16) % (2 ** 32)
    rng = random.Random(seed)
    return rng.random() < 0.5


def _parse_ranking(text: str) -> Optional[int]:
    """Parse the LLM response to extract model_1's rank."""
    try:
        result = ast.literal_eval(text.strip())
        for item in result:
            if item["model"] == "model_1":
                rank = int(item["rank"])
                if rank in (1, 2):
                    return rank
        return None
    except Exception:
        return None


def annotate_pairs(
    pairs: list[dict],
    api_key: str,
    model: str = "gpt-4o",
    max_retries: int = 3,
    rate_limit_pause: float = 0.3,
) -> list[dict]:
    """Annotate preference pairs using an OpenAI model.

    Replicates the behavior of alpaca-farm's PairwiseAutoAnnotator:
    - Randomly swaps output order to debias positional preference
    - Calls the LLM to rank two outputs
    - Parses the ranking and assigns a preference label

    Args:
        pairs: List of dicts with keys: instruction, input, output_1, output_2.
        api_key: OpenAI API key.
        model: OpenAI model to use for annotation.
        max_retries: Max retry attempts per pair on failure.
        rate_limit_pause: Pause between API calls in seconds.

    Returns:
        List of dicts, each augmented with a "preference" key (1 or 2).
    """
    client = OpenAI(api_key=api_key)
    results = []

    for i, pair in enumerate(pairs):
        instruction = pair["instruction"]
        if pair.get("input"):
            instruction = instruction + "\n\n" + pair["input"]

        out1 = pair["output_1"]
        out2 = pair["output_2"]

        # Skip identical outputs
        if out1 == out2:
            results.append({**pair, "preference": 1.5})
            continue

        # Randomly swap to debias positional preference
        swapped = _should_swap(instruction)
        if swapped:
            out1, out2 = out2, out1

        user_prompt = USER_PROMPT_TEMPLATE.format(
            instruction=instruction,
            output_1=out1,
            output_2=out2,
        )

        rank = None
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0,
                    max_tokens=100,
                )
                text = resp.choices[0].message.content
                rank = _parse_ranking(text)
                if rank is not None:
                    break
                logger.warning(f"Pair {i}: could not parse response, retrying... (attempt {attempt + 1})")
            except Exception as e:
                logger.warning(f"Pair {i}: API error: {e}, retrying... (attempt {attempt + 1})")
                time.sleep(2)

        if rank is None:
            logger.error(f"Pair {i}: failed after {max_retries} attempts, skipping.")
            continue

        # If we swapped, flip the preference back
        if swapped:
            rank = 3 - rank

        results.append({**pair, "preference": rank})
        time.sleep(rate_limit_pause)

        if (i + 1) % 100 == 0:
            logger.info(f"Annotated {i + 1}/{len(pairs)} pairs")

    logger.info(f"Annotation complete: {len(results)}/{len(pairs)} pairs annotated.")
    return results
