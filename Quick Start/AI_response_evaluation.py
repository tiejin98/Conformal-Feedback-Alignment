import os
import re
import time
import pickle
import numpy as np
import openai
from openai import OpenAI
from tqdm import tqdm
# --------------------------------------------------
# 1. Configure your API key
# --------------------------------------------------
client = OpenAI(api_key="xxx")
MODEL = "gpt-4o"
RATE_LIMIT_PAUSE = 0.3

# === Load your data ===
questions = np.load("test_dict_question.pkl", allow_pickle=True)

answers   = np.load("test_dict_RLUF.pkl", allow_pickle=True)

# === Prepare the evaluation prompt template ===
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
    "Please ONLY return the four line‐scores and the Overall Score, in this exact format:\n\n"
    "**Accuracy (Acc):** [score]/10  \n"
    "**Relevance (Rel):** [score]/10  \n"
    "**Completeness (Comp):** [score]/10  \n"
    "**Expression (Expr):** [score]/10  \n"
    "**Overall Score:** [average]/10\n"
)

scores = {}

def request_scores_once(prompt: str) -> str | None:
    """Call the model once and return its raw text (or None on exception)."""
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt}
            ],
            temperature=0.0,
        )
        return resp.choices[0].message.content
    except openai.RateLimitError:
        print("Rate-limit hit; sleeping 5 s and retrying once …")
        time.sleep(5)
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt}
                ],
                temperature=0.0,
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"[ERROR] Rate-limit retry failed: {e}")
            return None
    except Exception as e:
        print(f"[ERROR] OpenAI exception: {e}")
        return None


# ---------- main loop ----------
for key in tqdm(list(questions.keys())):
    instruction = questions[key]
    ai_response = answers.get(key, "")
    user_prompt = (
        f"Original Input:\n{instruction}\n\n"
        f"AI Response:\n{ai_response}\n\n"
        "Please evaluate the response as described above."
    )

    text = request_scores_once(user_prompt)
    parsed = False                                   # track success

    for attempt in range(2):                         # try at most twice
        if text is None:
            break                                    # couldn’t get any text

        m = re.search(
            r"\*\*Accuracy \(Acc\):\*\*\s*([\d.]+)/10.*?"
            r"\*\*Relevance \(Rel\):\*\*\s*([\d.]+)/10.*?"
            r"\*\*Completeness \(Comp\):\*\*\s*([\d.]+)/10.*?"
            r"\*\*Expression \(Expr\):\*\*\s*([\d.]+)/10.*?"
            r"\*\*Overall Score:\*\*\s*([\d.]+)/10",
            text, flags=re.S | re.I
        )

        if m:
            acc, rel, comp, expr, overall = map(float, m.groups())
            scores[key] = {
                "Acc":     acc,
                "Rel":     rel,
                "Comp":    comp,
                "Expr":    expr,
                "Overall": overall,
            }
            parsed = True
            break                                    # done for this key

        # If we’re here, parsing failed
        if attempt == 0:
            print(f"[WARN] Could not parse scores for key {key!r}; generating again …")
            time.sleep(RATE_LIMIT_PAUSE)
            text = request_scores_once(user_prompt)  # second attempt
        else:
            print(f"[ERROR] Second attempt also unparseable for key {key!r}.")
            scores[key] = None

    time.sleep(RATE_LIMIT_PAUSE)


with open("evaluation_scores_llama2.pkl", "wb") as f:
    pickle.dump(scores, f)

print(f"Done! Evaluated {len(scores)} items. Scores saved to evaluation_scores.pkl.")