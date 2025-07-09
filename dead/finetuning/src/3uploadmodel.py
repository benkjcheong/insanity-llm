from huggingface_hub import login, HfApi, upload_folder
import os
from dotenv import load_dotenv

# Load your Hugging Face token
load_dotenv()
hf_token = os.getenv("HuggingFace_Token")
login(token=hf_token)

repo_id = "bcben/manifesto-model"

# Push base LoRA model
upload_folder(
    repo_id=repo_id,
    folder_path="../lora_model",
    path_in_repo="lora_model",
    repo_type="model",
    token=hf_token
)