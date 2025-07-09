FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# 1. Install Python and minimal dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip curl wget && \
    rm -rf /var/lib/apt/lists/*

# 2. Upgrade pip and install prebuilt GPU-enabled llama-cpp-python
RUN pip3 install --upgrade pip && \
    pip3 install llama-cpp-python -v

# 3. Create app working directory and copy your app
WORKDIR /app
COPY app.py .

# 4. (Optional) Download Phi-2 GGUF model into container
RUN mkdir -p /app/models && \
    wget -O /app/models/phi-2.Q4_K_M.gguf \
    https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf

# 5. Run your Python app
CMD ["python3", "app.py"]
