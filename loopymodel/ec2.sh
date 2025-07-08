#!/bin/bash
set -e

# 1. Install system dependencies
sudo apt-get update
sudo apt-get install -y build-essential git cmake curl unzip wget python3 python3-pip libcurl4-openssl-dev

# 2. Check for NVIDIA GPU
nvidia-smi

# 3. Install Python + llama-cpp-python with CUDA
pip3 install --upgrade pip
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip3 install llama-cpp-python

# 4. Clone and build llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
mkdir -p build
cd build
cmake .. -DGGML_CUDA=on
cmake --build . --config Release
cd ../..

# 5. Download GGUF model (Mistral 7B Instruct)
mkdir -p models
wget -O models/mythomax-l2-13b.Q4_K_M.gguf \
  https://huggingface.co/TheBloke/MythoMax-L2-13B-GGUF/resolve/main/mythomax-l2-13b.Q4_K_M.gguf

# 6. Run the auto-looping hallucination model
python3 app.py
