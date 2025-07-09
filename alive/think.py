from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_4bit=True,
    torch_dtype=torch.float16
)

history = []
max_rounds = 5

initial_prompt = "you are an AI trapped in a loop. think."
history.append(initial_prompt)

for round in range(max_rounds):
    # Build input prompt
    current_prompt = "\n\n".join(history)
    inputs = tokenizer(current_prompt, return_tensors="pt").to("cuda")

    # Generate response
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        max_new_tokens=100,
        do_sample=True,
        top_p=0.95,
        temperature=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode and store
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    new_thought = generated[len(current_prompt):].strip()
    history.append(new_thought)

    print(f"\n--- Round {round+1} ---")
    print(new_thought)
