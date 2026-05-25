FROM python:3.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    IRODORI_HOST=0.0.0.0 \
    IRODORI_PORT=80 \
    IRODORI_MODEL_DEVICE=auto \
    IRODORI_CODEC_DEVICE=cuda

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        ffmpeg \
        git \
        libsndfile1 \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip uv

COPY pyproject.toml uv.lock README.md LICENSE THIRD_PARTY_NOTICES.md ./
COPY api ./api
COPY irodori_tts ./irodori_tts
COPY irodori_tts_lite ./irodori_tts_lite

RUN uv sync --frozen --no-dev \
    && . /app/.venv/bin/activate \
    && python -c "import torch; assert torch.version.cuda == '12.4', torch.version.cuda"

EXPOSE 80

CMD ["sh", "-c", ". /app/.venv/bin/activate && uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-80}"]
