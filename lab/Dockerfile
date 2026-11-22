# syntax=docker/dockerfile:1.7
# Build context = the repository root (see lab/docker-compose.yml); only pyproject.toml, src and lab/configs are used (lab/Dockerfile.dockerignore).
# Lab image: the ingestion pipeline + CLI, with the two embedding models BAKED IN, so ingestion needs no internet and
# 30 laptops do not download models at the same time. No OCR (the five lab files are born-digital).
FROM python:3.12-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY pyproject.toml ./
RUN mkdir -p src/orca_rag && touch src/orca_rag/__init__.py && pip install ".[ingest]"
COPY src ./src
RUN pip install --no-deps .
RUN FASTEMBED_CACHE_PATH=/models python -c "from fastembed import TextEmbedding, SparseTextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5'); SparseTextEmbedding('Qdrant/bm25')"

FROM python:3.12-slim AS runtime
RUN useradd --create-home --uid 10001 orca
COPY --from=build /opt/venv /opt/venv
COPY --from=build /models /models
COPY lab/configs /app/configs
WORKDIR /app
ENV PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    ORCA_HOME=/app ORCA_DATA_DIR=/data FASTEMBED_CACHE_PATH=/models HF_HUB_OFFLINE=1
RUN mkdir -p /data && chown -R orca:orca /data /app /models
USER orca
CMD ["python", "-m", "orca_rag.cli", "--help"]
