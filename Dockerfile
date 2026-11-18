# syntax=docker/dockerfile:1.7
# ---- build: resolve and install dependencies into a venv (cached layer; no source yet)
FROM python:3.12-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY pyproject.toml ./
RUN mkdir -p src/orca_rag && touch src/orca_rag/__init__.py && pip install ".[ingest]"
COPY src ./src
RUN pip install --no-deps .

# ---- runtime: no compilers, non-root, OCR optional
FROM python:3.12-slim AS runtime
ARG WITH_OCR=1
RUN if [ "$WITH_OCR" = "1" ]; then apt-get update && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng \
      && rm -rf /var/lib/apt/lists/*; fi \
 && useradd --create-home --uid 10001 orca
COPY --from=build /opt/venv /opt/venv
COPY configs /app/configs
WORKDIR /app
ENV PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    ORCA_HOME=/app ORCA_DATA_DIR=/data FASTEMBED_CACHE_PATH=/data/models HF_HOME=/data/hf
RUN mkdir -p /data && chown -R orca:orca /data /app
USER orca
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=4s --start-period=60s --retries=4 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=3).status==200 else 1)"
CMD ["uvicorn", "orca_rag.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080", "--proxy-headers"]
