"""Conformal prediction scoring and OpenAI evaluation helpers."""

import math
import logging

logger = logging.getLogger(__name__)


def compute_cp_score(dict_of_freq: dict, weight: float, weight_2: float,
                     similarity_model=None) -> tuple:
    """Compute conformal prediction nonconformity scores for each response.

    The score for each response is:
        score = 10 - (freq/total)*10 + (normalized_entropy/2)*weight
        For non-top responses: score -= similarity_to_top * weight_2

    Lower scores indicate higher confidence (more likely to be in prediction set).

    Args:
        dict_of_freq: Dict mapping response text to frequency count.
        weight: Weight for the entropy regularization term.
        weight_2: Weight for the similarity penalty term.
        similarity_model: Gensim FastText model for computing response similarity.

    Returns:
        Tuple of (score_dict, normalized_entropy).
    """
    dict_of_score = dict_of_freq.copy()
    total_frequency = sum(dict_of_freq.values())

    # Compute normalized entropy of the response distribution
    numerator = 0
    for key, value in dict_of_score.items():
        if total_frequency > 0:
            numerator += -(value / total_frequency) * math.log(value / total_frequency)
    if total_frequency <= 1:
        total_frequency = 2
    normalized_entropy = numerator / math.log(total_frequency)

    # Compute score for each response
    rank_1_response = ""
    for rank, (key, value) in enumerate(dict_of_score.items()):
        if rank == 0:
            rank_1_response = key
            dict_of_score[key] = 10 - value / total_frequency * 10 + normalized_entropy / 2 * weight
        else:
            dict_of_score[key] = 10 - value / total_frequency * 10 + normalized_entropy / 2 * weight
            if similarity_model is not None:
                try:
                    sim = similarity_model.wv.similarity(key, rank_1_response)
                    dict_of_score[key] -= sim * weight_2
                except KeyError:
                    pass

    return dict_of_score, normalized_entropy


def get_openai_score(client, prompt: str, generated_summary: str,
                     reference_summary: str, model: str = "gpt-4o") -> float:
    """Score a generated summary using OpenAI API.

    Args:
        client: OpenAI client instance.
        prompt: Original text that was summarized.
        generated_summary: The generated summary to evaluate.
        reference_summary: Ground truth reference summary.
        model: OpenAI model to use for scoring.

    Returns:
        Score between 0 and 1.
    """
    score_prompt = f"""
You are an experienced evaluator tasked with scoring a generated summary. Here are the inputs:

1. Original Text:
{prompt}

2. Reference Answer:
{reference_summary}

3. Generated Answer:
{generated_summary}

Please evaluate the generated answer and assign a score between 0 and 1, where 0 indicates it completely fails to meet the standards, and 1 indicates it fully meets the standards. Use the following criteria:

- Completeness: Does the generated answer cover all the essential information provided in the reference answer?
- Accuracy: Is the information in the generated answer consistent with the original text, without factual errors?
- Conciseness: Is the generated answer succinct, conveying the main ideas without unnecessary detail or irrelevant content?
- Language Quality: Is the generated answer clear and fluent, with correct grammar and appropriate word usage?

Output your result strictly in the following format:

# -------------------------------
Score: <value between 0 and 1>
"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": score_prompt}],
        )
        result = response.choices[0].message.content
        score_line = result.split("\n")[0]
        score = float(score_line.split(": ")[1])
        return score
    except Exception as e:
        logger.warning(f"Error in OpenAI API call: {e}")
        return 0.0
