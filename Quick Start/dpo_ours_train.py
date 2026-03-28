import os
import torch
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOConfig
from cfa.models.weighted_dpo import WeightedDPOTrainer

# Load API keys from .env
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
hf_token = os.environ.get("HF_TOKEN")

# Load the new DPO-format dataset
dataset = load_dataset("json", data_files="dpo_data_llama2_withuncertainty.json", split="train")

# Initialize model and tokenizer
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, token=hf_token)

# Create a config for DPO
config = DPOConfig(output_dir="/home/Model/Llama2-7b-RLUF", per_device_train_batch_size=2, learning_rate=1.5e-6,
                   logging_steps=1500,save_steps=1500,save_total_limit=2)

# Initialize the DPOTrainer
trainer = WeightedDPOTrainer(
    model=model,
    tokenizer=tokenizer,
    args=config,
    train_dataset=dataset,
)

# Start training
trainer.train()
