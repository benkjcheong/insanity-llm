from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Base model hosted on HF (can be quantized)
base_model = AutoModelForCausalLM.from_pretrained(
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    device_map="auto",
    load_in_4bit=True
)
tokenizer = AutoTokenizer.from_pretrained("unsloth/mistral-7b-instruct-v0.3-bnb-4bit")

# Load LoRA adapter from HF
model = PeftModel.from_pretrained(base_model, "bcben/manifesto-model")

# Run inference
prompt = "//"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=100)
    print(tokenizer.decode(output[0], skip_special_tokens=True))