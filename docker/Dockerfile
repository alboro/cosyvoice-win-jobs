FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        git-lfs \
        ffmpeg \
        sox \
        libsndfile1 \
        libsndfile1-dev \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone CosyVoice with third-party submodules (Matcha-TTS).
# This layer is stable unless the upstream repo changes.
RUN git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git vendor/CosyVoice

# Install pynini first via pip wheel (Python 3.10, linux x86_64).
# If PyPI wheel is unavailable, install from conda-forge as fallback.
RUN pip install --no-cache-dir "pynini==2.1.6" || \
    pip install --no-cache-dir pynini

# Install ttsfrd from the bundled wheel in vendor (pre-compiled, platform-specific).
# This is optional — text_frontend=False (our Russian default) does not need it,
# but installing it avoids import errors when the runtime probes the package.
RUN find vendor/CosyVoice -name "ttsfrd*.whl" -print0 2>/dev/null | \
    xargs -0 -r pip install --no-cache-dir; true

# Install remaining CosyVoice requirements.
# Run from the vendor dir so relative file:// paths in requirements.txt resolve.
RUN cd vendor/CosyVoice && pip install --no-cache-dir -r requirements.txt

# ── Bake model weights (separate layer — only rebuilt when model version changes) ──
# Fun-CosyVoice3-0.5B-2512 is the Dec-2025 release of CosyVoice3 with Russian support.
RUN pip install --no-cache-dir huggingface_hub hf_transfer && \
    HF_HUB_ENABLE_HF_TRANSFER=1 python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download( \
    'FunAudioLLM/Fun-CosyVoice3-0.5B-2512', \
    local_dir='/app/pretrained_models/Fun-CosyVoice3-0.5B', \
    local_dir_use_symlinks=False, \
    ignore_patterns=['*.gitattributes', 'README*', '*.md'] \
)"

# ── Wrapper source (thin layer — rebuilt on every code push) ──
COPY pyproject.toml ./
COPY src/ src/
RUN pip install --no-cache-dir -e .

# Runtime defaults (all overridable via env or CLI flags)
ENV COSYVOICE_HOST=0.0.0.0 \
    COSYVOICE_PORT=8040 \
    COSYVOICE_MODEL_ID=Fun-CosyVoice3-0.5B \
    COSYVOICE_MODEL_DIR=/app/pretrained_models/Fun-CosyVoice3-0.5B \
    COSYVOICE_SHARED_DIR=/app/shared \
    COSYVOICE_JOBS_DIR=/app/.data/jobs \
    COSYVOICE_VOICES_DIR=/app/.data/voices \
    PYTHONPATH=/app/src:/app/vendor/CosyVoice:/app/vendor/CosyVoice/third_party/Matcha-TTS

EXPOSE 8040

CMD ["python", "-m", "cosyvoice_win.server"]
