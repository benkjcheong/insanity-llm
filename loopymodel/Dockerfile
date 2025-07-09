FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Install build tools and dependencies
RUN apt-get update && apt-get install -y \
    git build-essential cmake curl wget python3 python3-pip \
    libcurl4-openssl-dev libopenblas-dev ninja-build && \
    rm -rf /var/lib/apt/lists/*

# Set environment flags for CUDA build
ENV CMAKE_ARGS="-DGGML_CUDA=on"

# Install llama-cpp-python with CUDA support
RUN pip3 install --upgrade pip && \
    pip3 install llama-cpp-python

# 3. Clone llama.cpp
RUN git clone https://github.com/ggerganov/llama.cpp /llama.cpp && \
    cd /llama.cpp && mkdir build && cd build && \
    cmake .. -DGGML_CUDA=on && cmake --build . --config Release -- -j$(nproc)

# 4. Create app working directory
WORKDIR /app

# 5. Copy your application script
COPY app.py /app/

# 6. Download Phi-2 GGUF model
RUN mkdir -p /app/models && \
    wget -O /app/models/phi-2.Q4_K_M.gguf \
    https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf

# 7. Run your app
CMD ["python3", "app.py"]
