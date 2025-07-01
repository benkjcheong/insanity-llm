from datasets import Dataset
from huggingface_hub import login
from dotenv import load_dotenv
from striprtf.striprtf import rtf_to_text
import pdfplumber
import os

# Load Hugging Face token
load_dotenv()
hf_token = os.getenv("HuggingFace_Token")

input_folder = "../dataset"
output_folder = "../txt_datasets"
os.makedirs(output_folder, exist_ok=True)

# 1. Extract and save text from PDFs and RTFs
for filename in os.listdir(input_folder):
    file_path = os.path.join(input_folder, filename)
    full_text = ""

    if filename.endswith(".pdf"):
        try:
            with pdfplumber.open(file_path) as pdf:
                full_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

    elif filename.endswith(".rtf"):
        try:
            try:
                with open(file_path, "r", encoding="utf-8") as rtf_file:
                    rtf_content = rtf_file.read()
            except UnicodeDecodeError:
                with open(file_path, "r", encoding="cp1252") as rtf_file:
                    rtf_content = rtf_file.read()
            full_text = rtf_to_text(rtf_content)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

    # Save to .txt if extracted
    if full_text.strip():
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        txt_path = os.path.join(output_folder, txt_filename)
        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(full_text)
        except Exception as e:
            print(f"Error writing {txt_filename}: {e}")
            continue

# 2. Load text back from /txt_datasets
texts = []
for filename in os.listdir(output_folder):
    file_path = os.path.join(output_folder, filename)
    if filename.endswith(".txt"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    texts.append({"text": content})
        except Exception as e:
            print(f"Error reading {filename}: {e}")

# 3. Push to Hugging Face if anything collected
if texts:
    dataset = Dataset.from_list(texts)
    login(token=hf_token)
    dataset.push_to_hub("bcben/manifestos")
else:
    print("No readable text extracted. Nothing to upload.")
