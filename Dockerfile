# CUDA runtime base: the torch wheels below are built against cu12x, and the
# *runtime* variant (not devel) keeps the image small enough to pull onto a
# GPUStack worker. Override with --build-arg CUDA=12.4.1 if the driver is older.
ARG CUDA=12.6.1
FROM nvidia/cuda:${CUDA}-cudnn-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.11
# Match this to CUDA: cu126 for 12.6, cu124 for 12.4, and so on.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu126

# ffmpeg lets us decode containers libsndfile cannot (m4a, webm/opus). The rest
# are build/runtime basics; libsndfile1 is what soundfile binds to.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        ca-certificates \
        curl \
        git \
        build-essential \
        ffmpeg \
        libsndfile1 \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-venv \
        python${PYTHON_VERSION}-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
    && python${PYTHON_VERSION} /tmp/get-pip.py \
    && rm /tmp/get-pip.py \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python3

WORKDIR /app

# Install the CUDA build of torch first so pip does not resolve the CPU wheel
# from PyPI. This layer is the slow one and rarely changes.
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir torch --index-url ${TORCH_INDEX}

COPY pyproject.toml README.md ./
COPY app ./app

# kenlm has no wheel and compiles from source; build-essential covers it.
ARG INSTALL_LM=true
RUN if [ "$INSTALL_LM" = "true" ]; then \
        python -m pip install --no-cache-dir ".[lm]"; \
    else \
        python -m pip install --no-cache-dir .; \
    fi

# Cache the Hugging Face downloads (model + EmbeddingGemma encoder) outside the
# image so containers do not re-download them.
ENV HF_HOME=/cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=qnguyen3/QUAN-Transcribe-1.0 \
    SERVED_MODEL_NAME=QUAN-Transcribe-1.0 \
    HOST=0.0.0.0 \
    PORT=8000 \
    PRELOAD=true \
    LOG_LEVEL=info \
    OMP_NUM_THREADS=8

# Lightweight healthcheck: /health reports real model state but stays cheap, so
# it is safe to run during a multi-minute weight download.
HEALTHCHECK --interval=30s --timeout=5s --start-period=900s --retries=5 \
    CMD curl -fsS http://127.0.0.1:${PORT}/health || exit 1

EXPOSE 8000

# The bare entrypoint is what GPUStack appends {{model_path}}/{{port}} style
# arguments to. `--serve` is accepted for command parity with other backends.
ENTRYPOINT ["quan-asr", "--serve"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
