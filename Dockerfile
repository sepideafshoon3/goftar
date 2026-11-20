# Goftar in a container: the web UI on port 7860 (or the REST API, or the CLI).
#
#   CPU:  docker build -t goftar .
#   GPU:  docker build -t goftar:gpu --build-arg GPU=1 .   (needs the NVIDIA Container Toolkit)
#
#   docker run --rm -p 7860:7860 -v goftar-models:/data/models -v goftar-cache:/data/cache goftar
#   docker run --rm --gpus all -p 7860:7860 -v goftar-models:/data/models goftar:gpu
#
# The model is not in the image: it is downloaded into the goftar-models volume on the first
# start, and every later start runs with --offline.
FROM python:3.11-slim

ARG GPU=0
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    HF_HUB_DISABLE_TELEMETRY=1 \
    GOFTAR_MODEL=large-v3-turbo \
    # CUDA libraries installed by the "gpu" extra (pip wheels); harmless on the CPU image
    LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE /tmp/goftar/
COPY src /tmp/goftar/src
RUN extras="api,ui"; if [ "$GPU" = "1" ]; then extras="$extras,gpu"; fi \
    && pip install "/tmp/goftar[$extras]" \
    && rm -rf /tmp/goftar

RUN useradd --create-home --uid 1000 goftar \
    && mkdir -p /data/models /data/cache \
    && chown -R goftar:goftar /data
COPY --chown=goftar:goftar docker/config.yaml /data/goftar.yaml
COPY docker/entrypoint.sh /usr/local/bin/goftar-entrypoint

USER goftar
WORKDIR /data
VOLUME ["/data/models", "/data/cache"]
EXPOSE 7860 8000

ENTRYPOINT ["goftar-entrypoint"]
CMD ["ui", "--host", "0.0.0.0", "--port", "7860"]
