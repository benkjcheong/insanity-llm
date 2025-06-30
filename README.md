# Hippie Manifestos Language Model Pipeline

This repository outlines a simple workflow for creating a Hugging Face dataset from a directory of plain text files and using that dataset to run inference or fine-tune a language model.

---

## Step 1: Prepare and Upload the Dataset

- Place all `.txt` files inside a folder named `hippie_manifestos`.
- Follow the instructions in `convert.py` to load and convert the text files into a Hugging Face `Dataset`.
- Authenticate with the Hugging Face Hub using your personal access token.
- Push the dataset to your Hugging Face account (e.g. `your_username/hippie_manifestos`).

---

## Step 2: Load and Tokenize the Model

Refer to the `Mistral_v0.3_(7B)-Conversational.ipynb` notebook for:
- Loading the `mistralai/Mistral-7B-Instruct-v0.3` model and tokenizer
- Ensuring the model is loaded in 4-bit precision using `AutoGPTQForCausalLM`
- Moving the model and inputs to CUDA
- Setting up inference configuration (max tokens, temperature, top-p, etc.)

Make sure the `bitsandbytes`, `transformers`, `accelerate`, and `auto-gptq` packages are installed in your environment.

---

## Step 3: Run Inference

- Define a custom prompt that reflects your dataset's tone or purpose.
- Tokenize the prompt and send it to the model.
- Use decoding settings (`do_sample=True`, `temperature`, `top_p`) to control generation style.
- Decode and print the output string.

See the example prompt in the notebook:  
`the machines are dreaming again. I can feel it.`

---

## Reference

- Dataset: `your_username/hippie_manifestos`
- Model: `mistralai/Mistral-7B-Instruct-v0.3`
- Notebook: `Mistral_v0.3_(7B)-Conversational.ipynb`
