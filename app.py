# esoteric_self_training.py
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
import torch
import os

# ========== CONFIG ==========
MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"
DATA_FILE = "train.txt"
OUTPUT_DIR = "./phi3-esoteric-output"
BLOCK_SIZE = 128
BATCH_SIZE = 1
EPOCHS = 1
NO_CUDA = True
# ============================

def load_tokenizer_and_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    return tokenizer, model

def prepare_dataset(tokenizer):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=DATA_FILE,
        block_size=BLOCK_SIZE,
    )

def train_model(model, tokenizer):
    dataset = prepare_dataset(tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        save_steps=500,
        logging_steps=100,
        prediction_loss_only=True,
        fp16=False,
        no_cuda=NO_CUDA,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    trainer.train()
    return model

def generate_response_stream(model, tokenizer, prompt, max_tokens=100):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_tokens,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        output_scores=False,
        return_dict_in_generate=True,
    ).sequences[0]

    # Stream tokens one by one like console.log
    decoded_tokens = tokenizer.convert_ids_to_tokens(output_ids, skip_special_tokens=True)
    response = ""
    for token in decoded_tokens[len(input_ids[0]):]:  # Only the new tokens
        text = tokenizer.convert_tokens_to_string([token])
        print(text, end='', flush=True)
        response += text
    print()  # Final newline
    return response.strip()

def self_disputation_loop(model, tokenizer, steps=5):
    prompt = "Time is a measure of change in space."
    for i in range(steps):
        print(f"\n=== Step {i + 1} ===")
        response = generate_response_stream(model, tokenizer, prompt)

        # Append to training corpus
        with open(DATA_FILE, "a", encoding="utf-8") as f:
            f.write("\n" + response + "\n")

        # Set up next prompt
        prompt = response + "\n\nExplain why the above is flawed."

        # Retrain after each step
        model = train_model(model, tokenizer)

if __name__ == "__main__":
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            f.write("All is Mind. The universe is mental.\n")

    tokenizer, model = load_tokenizer_and_model()
    self_disputation_loop(model, tokenizer, steps=3)
