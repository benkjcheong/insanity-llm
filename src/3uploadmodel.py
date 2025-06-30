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
    folder_path="lora_model",
    path_in_repo="lora_model",
    repo_type="model",
    token=hf_token
)

# Push merged FP16 model
upload_folder(
    repo_id=repo_id,
    folder_path="merged_f16",
    path_in_repo="merged_f16",
    repo_type="model",
    token=hf_token
)

# Push GGUF quantized model
upload_folder(
    repo_id=repo_id,
    folder_path="gguf_q4km",
    path_in_repo="gguf_q4km",
    repo_type="model",
    token=hf_token
)