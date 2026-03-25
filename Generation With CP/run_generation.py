import torch
import time
import math
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer,AutoTokenizer,AutoModelForCausalLM
from datasets import load_dataset
from collections import defaultdict
import json
import numpy as np
from gensim.test.utils import common_texts, datapath
from gensim.models import FastText
from tqdm import tqdm
import pickle
from openai import OpenAI

client = OpenAI(api_key="xxxx")




def get_openai_score(prompt, generated_summary, reference_summary):
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
            model="gpt-4o",
            messages=[
                {"role": "user", "content": score_prompt},
            ]
        )
        result = response.choices[0].message.content
        score_line = result.split("\n")[0]
        score = float(score_line.split(": ")[1])
        return score
    except Exception as e:
        print(f"Error in OpenAI API call: {e}")
        return 0.0




# -------------------------------
device_name = [f"cuda:0"]

calibration_size = 50
test_size = None
kshot = 32
num_return_sequences = 1
sampling_num = 60
alpha = 0.4
np.random.seed(42)
torch.manual_seed(42)
model_name = "/home/Model/llama2-sft/checkpoints-1500"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
model.to(device_name[0])
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
similarity_model = FastText(sentences=common_texts, vector_size=200, min_count=1)


dataset = load_dataset("openai/summarize_from_feedback", "axis")
val_data = dataset["validation"]

unique_data = {}
for example in val_data:
    info = example["info"]
    summary = example["summary"]['text']
    this_id = info["id"]

    if this_id not in unique_data:
        unique_data[this_id] = (info["post"], summary)
all_data = list(unique_data.items())

questions = []
answers = []
test_questions = []
test_answers = []

divide_bar = 0
num = 0

for (the_id, (the_post, the_summary)) in all_data:
    divide_bar += 1
    if divide_bar < 20000:
        words = the_post.split()
        if len(words) <= 500:
            num += 1
            if num <= calibration_size:
                question_text = "Please summarize this text:\n" + the_post + "\nSummary:"
                questions.append(question_text)
                answers.append(the_summary)
            else:
                test_question_text = "Please summarize this text:\n" + the_post+ "\nSummary:"
                test_questions.append(test_question_text)
                test_answers.append(the_summary)

test_size = len(test_questions)
print(f"Collected {len(questions)} calibration samples, {test_size} test samples.")



def few_shot(kshot):
    return ""

def generate_text(
    prompt,
    do_sample=True,
    max_new_tokens=128,
    num_return_sequences=num_return_sequences,
    return_dict_in_generate=True,
    output_scores=True
):
    inputs = tokenizer(prompt, return_tensors="pt").to(device_name[0])
    attention_mask = inputs['attention_mask']

    outputs = model.generate(
        inputs.input_ids,
        do_sample=do_sample,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=return_dict_in_generate,
        output_scores=output_scores,
        temperature=0.35,
    )

    transition_scores = model.compute_transition_scores(
        outputs.sequences, outputs.scores, normalize_logits=True
    )
    return outputs, transition_scores

prompt = few_shot(kshot)
# #
generated_responses = []
accuracy_scores = []

print("\n=== Generating on Calibration Set ===")
for i in tqdm(range(calibration_size)):
    idx_q = i + kshot if (i + kshot) < len(questions) else i
    prompt_question = questions[idx_q]
    reference_answer = answers[idx_q]
    final_prompt = prompt + prompt_question  # 这里 = prompt_question
    response_dict = defaultdict(int)

    for sampling_index in range(sampling_num):
        response, score = generate_text(final_prompt)
        full_text = tokenizer.decode(response.sequences[0], skip_special_tokens=False)

        idx_found = full_text.find(final_prompt)
        if idx_found != -1:
            summary_text = full_text[idx_found + len(final_prompt):].strip()
        else:
            summary_text = full_text.strip()

        summary_text = summary_text.lower()
        response_dict[summary_text] += 1
    # Compute GPT scores for unique summaries
    response_scores = {}
    for summary_text in response_dict.keys():
        openai_score = get_openai_score(prompt_question, summary_text, reference_answer)
        response_scores[summary_text] = openai_score
    
    generated_responses.append(dict(response_dict))
    accuracy_scores.append(dict(response_scores))


# calibration set: 写 generation.txt
with open("generation_llama2.txt", "w", encoding="utf-8") as file:
    file.write(str(generated_responses))
    file.write("\n")

with open("generation_llama2_accuracy.txt", "w", encoding="utf-8") as file_acc:
    file_acc.write(json.dumps(accuracy_scores, indent=2))
    file_acc.write("\n")

# start_idx = kshot
# end_idx = kshot + calibration_size
# if end_idx > len(answers):
#     end_idx = len(answers)
# calib_answers = answers[start_idx:end_idx]
#
# with open("answers_llama2.txt", "w", encoding="utf-8") as file2:
#     file2.write(str(calib_answers))
#     file2.write("\n")

print("Calibration set generation done")


response_dict_final = {}
print("\n=== Generating on Test Set ===")
with open("generation_test_llama2.txt", "w", encoding="utf-8") as file:
    for k in tqdm(range(test_size)):
        final_prompt_test = prompt + test_questions[k]

        response_dict_test = defaultdict(int)
        for sampling_index in range(sampling_num):
            response, score = generate_text(final_prompt_test)
            full_text = tokenizer.decode(response.sequences[0], skip_special_tokens=False)

            idx_found = full_text.find(final_prompt_test)
            if idx_found != -1:
                test_summary_text = full_text[idx_found + len(final_prompt_test):].strip()
            else:
                test_summary_text = full_text.strip()

            test_summary_text = test_summary_text.lower()
            response_dict_test[test_summary_text] += 1

        response_dict_test = dict(response_dict_test)
        response_dict_final[final_prompt_test] = response_dict_test
        file.write(str(response_dict_test))
        file.write("\n")

with open('response_dict_llama2.pkl', 'wb') as f:
    pickle.dump(response_dict_final, f)
print(response_dict_final)
print("Done Test")
