from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch

dataset = load_dataset("openai/summarize_from_feedback", "axis")
val_data = dataset["validation"]
access = "xxxxxxx"

unique_data = {}
for example in val_data:
    info = example["info"]           # contains keys "id", "post", etc.
    summary = example["summary"]['text']   # Ground Truth Summary
    this_id = info["id"]
    if this_id not in unique_data:   # only keep the first occurrence
        unique_data[this_id] = (info["post"], summary)

all_data = list(unique_data.items())

data_list = []
for _id, (post, summary) in all_data:
    prompt = "Please summarize this text:\n" + post + "\nSummary:"
    # We'll concatenate the prompt with the summary.
    full_text = prompt + " " + summary
    data_list.append({
        "prompt": prompt,
        "target": summary,
        "text": full_text  # full text used for tokenization
    })

train_dataset = Dataset.from_list(data_list)

model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False,token=access)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto",torch_dtype=torch.bfloat16,cache_dir="/home/Model",token=access)

def tokenize_and_mask(example):
    # Tokenize the full text (prompt + target)
    full = tokenizer(example["text"], truncation=True, max_length=1024)
    # Tokenize the prompt separately.
    prompt_tokens = tokenizer(example["prompt"], truncation=True, max_length=1024)["input_ids"]
    full_ids = full["input_ids"]
    # Create labels: set tokens corresponding to prompt to -100 so that loss is computed only on target.
    labels = full_ids.copy()
    prefix_length = len(prompt_tokens)
    for i in range(prefix_length):
        labels[i] = -100
    full["labels"] = labels
    return full

tokenized_dataset = train_dataset.map(tokenize_and_mask, batched=False, remove_columns=["prompt", "target", "text"])

def custom_data_collator(features):
    batch = tokenizer.pad(
        {"input_ids": [f["input_ids"] for f in features],
         "attention_mask": [f["attention_mask"] for f in features]},
        return_tensors="pt"
    )
    # Then, pad the labels manually using -100.
    max_length = batch["input_ids"].shape[1]
    padded_labels = []
    for f in features:
        labels = f["labels"]
        pad_length = max_length - len(labels)
        padded_labels.append(labels + [-100] * pad_length)
    batch["labels"] = torch.tensor(padded_labels)
    return batch

training_args = TrainingArguments(
    output_dir="/home/Model/llama2-sft",
    num_train_epochs=5,
    per_device_train_batch_size=2,    # adjust based on your GPU memory
    learning_rate=2e-5,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    bf16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=custom_data_collator,
)

trainer.train()
