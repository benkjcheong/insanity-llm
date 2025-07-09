import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import threading

MODEL_NAME = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    load_in_4bit=True,
    torch_dtype=torch.float16
)

def generate_streamed_response(prompt: str, temperature=0.7, top_p=0.95, max_new_tokens=300) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    thread = threading.Thread(target=model.generate, kwargs=dict(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        streamer=streamer
    ))
    thread.start()

    output = ""
    for token in streamer:
        print(token, end="", flush=True)
        output += token
    print()
    return output.strip()

def build_prompt(role_instruction: str, history: list[str], topic: str, speaker: str) -> str:
    recent_history = history[-6:] if len(history) > 6 else history
    dialogue = "\n".join(recent_history + [f"{speaker}: Please challenge the previous statement."])
    return f"{role_instruction}\nDebate Topic: {topic}\n\n{dialogue}"

def debate(topic: str):
    history = []
    last_line = topic
    turn_count = 0

    # Conflict-focused instructions
    agent_a_instruction = (
        "You are Agent A. You are confident, logical, and combative.\n"
        "You are debating Agent B. Your goal is to find flaws in their logic and rebut their claims.\n"
        "Never agree outright unless absolutely necessary. Start arguments, challenge assumptions, and stay on topic."
    )

    agent_b_instruction = (
        "You are Agent B. You are skeptical, sharp, and aggressive.\n"
        "You are debating Agent A. Your job is to dismantle Agent A’s points and expose logical weaknesses.\n"
        "Avoid concession unless their logic is flawless. Contradict constructively and critically."
    )

    print(f"Debate Topic: {topic}\n")

    while True:
        turn_count += 1
        temp = 0.7 + (0.01 * (turn_count // 10))  # gradually add variation over time

        # Agent A turn
        print("Agent A: ", end="", flush=True)
        prompt_a = build_prompt(agent_a_instruction, history + [f"Agent B: {last_line}"], topic, speaker="Agent A")
        response_a = generate_streamed_response(prompt_a, temperature=temp)
        history.append(f"Agent A: {response_a}")

        # Agent B turn
        print("Agent B: ", end="", flush=True)
        prompt_b = build_prompt(agent_b_instruction, history, topic, speaker="Agent B")
        response_b = generate_streamed_response(prompt_b, temperature=temp)
        history.append(f"Agent B: {response_b}")

        last_line = response_b

if __name__ == "__main__":
    debate("Should illegal immigration be allowed?")
