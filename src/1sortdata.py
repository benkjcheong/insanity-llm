from datasets import Dataset
from huggingface_hub import login
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import os

load_dotenv()
hf_token = os.getenv("HuggingFace_Token")

folder = "dataset"
texts = []

for filename in os.listdir(folder):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(folder, filename)
        reader = PdfReader(pdf_path)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() or ""
        texts.append({"text": full_text})

dataset = Dataset.from_list(texts)

login(token=hf_token)
dataset.push_to_hub("bcben/manifestos")
