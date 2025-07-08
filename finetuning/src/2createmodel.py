# train_manifesto_model.py
import os
import torch
from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer

# ---- Load model ----
max_seq_length = 2048
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# ---- Load and preprocess dataset ----
streamed_dataset = load_dataset("bcben/manifestos", split="train", streaming=True)

chunk_size = 1000
max_docs = 100
chunks = []

for i, item in enumerate(streamed_dataset):
    text = item["text"]
    chunks.extend([{"text": text[j:j+chunk_size]} for j in range(0, len(text), chunk_size)])
    if i + 1 >= max_docs:
        break

dataset = Dataset.from_list(chunks)
print("Example:", dataset[0])

# ---- Train model ----
start_gpu_memory = torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024
max_memory = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
    ),
)

trainer_stats = trainer.train()

# ---- Training stats ----
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# ---- Inference ----
prompt = "the machines are dreaming again. I can feel it.\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(
    input_ids=inputs["input_ids"],
    max_new_tokens=100,
    do_sample=True,
    temperature=0.95,
    top_p=0.9
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# ---- Save models ----
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

model.save_pretrained_merged("merged_f16", tokenizer, save_method="merged_16bit")
model.save_pretrained_gguf("gguf_q4km", tokenizer, quantization_method="q4_k_m")
