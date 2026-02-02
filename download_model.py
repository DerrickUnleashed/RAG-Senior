from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
SAVE_DIR = "./models/mistral-7b-instruct-v0.2"

print("Starting download...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    use_fast=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)

tokenizer.save_pretrained(SAVE_DIR)
model.save_pretrained(SAVE_DIR)

print(f"✅ Model downloaded and saved to: {SAVE_DIR}")
