import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer (can be shared or different)
MODEL_NAME = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    load_in_4bit=True,
    torch_dtype=torch.float16
)

def generate_response(prompt: str, max_new_tokens: int = 200) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

def clean_response(full_response: str, prompt: str) -> str:
    return full_response.replace(prompt.strip(), "").strip()

def agents_agree(statement: str) -> bool:
    agreement_markers = ["i agree", "you are right", "that makes sense", "we are aligned"]
    return any(phrase in statement.lower() for phrase in agreement_markers)

def debate(prompt: str, max_rounds: int = 10):
    history = []
    system_prefix_a = "You are Agent A. Be logical and assertive.\n"
    system_prefix_b = "You are Agent B. Be critical but open-minded.\n"

    last_statement = prompt
    print(f"🎯 Debate Topic: {prompt}\n")

    for round_num in range(max_rounds):
        # Agent A responds
        input_a = system_prefix_a + "\n".join(history + [f"Agent B: {last_statement}", "Agent A:"])
        response_a = clean_response(generate_response(input_a), input_a)
        print(f"🅰️ A: {response_a}")
        history.append(f"Agent A: {response_a}")
        if agents_agree(response_a):
            print("\n✅ Agent A agrees — debate concluded.")
            return

        # Agent B responds
        input_b = system_prefix_b + "\n".join(history + ["Agent B:"])
        response_b = clean_response(generate_response(input_b), input_b)
        print(f"🅱️ B: {response_b}")
        history.append(f"Agent B: {response_b}")
        if agents_agree(response_b):
            print("\n✅ Agent B agrees — debate concluded.")
            return

        last_statement = response_b

    print("\n❌ Max rounds reached — no agreement.")

if __name__ == "__main__":
    initial_prompt = "Should artificial intelligence be granted legal personhood?"
    debate(initial_prompt, max_rounds=8)
