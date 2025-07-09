FROM nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu22.04

# 1. System dependencies
RUN apt-get update && apt-get install -y \
    git build-essential cmake curl wget python3 python3-pip \
    libcurl4-openssl-dev && rm -rf /var/lib/apt/lists/*

# 2. Install Python packages
RUN pip3 install --upgrade pip && \
    CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip3 install llama-cpp-python

# 3. Clone llama.cpp
RUN git clone https://github.com/ggerganov/llama.cpp /llama.cpp && \
    cd /llama.cpp && mkdir build && cd build && \
    cmake .. -DGGML_CUDA=on && cmake --build . --config Release -- -j$(nproc)

# 4. Create app entrypoint
WORKDIR /app
COPY app.py /app/
COPY models /app/models

CMD ["python3", "app.py"]
