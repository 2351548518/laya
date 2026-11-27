# Build and runtime share one base so the copied virtualenv matches its interpreter.
ARG PYTHON_IMAGE=python:3.11-slim-trixie

FROM ${PYTHON_IMAGE} AS build

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# The virtualenv is copied into the runtime image. Upgrade its bundled installers
# so an up-to-date base does not still ship ensurepip's older pip/setuptools (#739).
RUN python -m venv --upgrade-deps /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# CPU by default on AMD64 and ARM64. The CUDA and DGX Spark overlays select
# their wheel index. Verify the final environment after resolving the app extras.
ARG TORCH_INDEX=cpu
ARG TORCH_VERSION=2.14.0
COPY docker/check_torch.py /opt/check_torch.py
RUN pip install "torch==${TORCH_VERSION}" torchvision --index-url "https://download.pytorch.org/whl/${TORCH_INDEX}" \
    && python /opt/check_torch.py "${TORCH_INDEX}"

WORKDIR /src
COPY pyproject.toml setup.py README.md LICENSE ./
COPY laya/ ./laya/
# The `serve` extra puts `laya-serve` (POST /v1/systemone, GET /health) in the image, so
# the same image can run a one-shot request or serve the Jev-compatible API.
RUN pip install ".[serve,multimodal]" "torch==${TORCH_VERSION}" \
    && python /opt/check_torch.py "${TORCH_INDEX}" \
    && python -c 'import laya.multimodal, torchvision' \
    && pip check

FROM ${PYTHON_IMAGE} AS runtime

LABEL org.opencontainers.image.title="laya-multimodal" \
      org.opencontainers.image.source="https://github.com/2351548518/laya" \
      org.opencontainers.image.licenses="Apache-2.0"

# torch 2.14 swaps some eager CUDA ops (bmm, topk, sum, norms) for Triton kernels that it
# compiles on the first inference, which needs a C compiler this image does not carry: the
# container reports healthy, then every request fails (#365). The stock kernels give the same
# answers at the same latency.
ENV PATH="/opt/venv/bin:$PATH" \
    TORCH_DISABLE_NATIVE_JIT=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    USE_TF=0 \
    USE_TORCH=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=4 \
    LAYA_DEVICE=cpu \
    HF_HOME=/home/laya/.cache/huggingface

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 10001 laya \
    && useradd --uid 10001 --gid laya --create-home laya \
    && mkdir -p /home/laya/.cache/huggingface \
    && chown -R laya:laya /home/laya/.cache

COPY --from=build /opt/venv /opt/venv
COPY LICENSE /usr/share/doc/laya/LICENSE
COPY examples/docker/ /opt/laya/examples/
COPY docker/entrypoint.py /opt/laya/entrypoint.py
USER laya
WORKDIR /home/laya

ENTRYPOINT ["python", "/opt/laya/entrypoint.py"]
CMD ["python", "/opt/laya/examples/quickstart.py"]
