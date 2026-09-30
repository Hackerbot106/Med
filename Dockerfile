# Healthcare Triage Assistant - production container image.
#
# Build:  docker build -t triage-assistant .
# Run:    docker compose up   (see docker-compose.yml for volumes/env)
#
# The local AI model weights (~1GB, optionally +~2GB for the vision model)
# are NOT baked into this image - they're fetched once into a mounted
# `models/` volume via download_model.py, so the image stays small and
# rebuilding the app doesn't mean re-downloading model weights, and the
# same weights can be shared across multiple container restarts/upgrades.

FROM python:3.11-slim AS base

# System dependencies: tesseract-ocr + poppler-utils for the OCR pipeline
# (triage_engine/ocr.py); build-essential is needed for llama-cpp-python's
# CPU build on platforms without a prebuilt wheel available.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Non-root runtime user - never run a web-facing process as root.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/data /app/models /app/logs \
    && chown -R appuser:appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1 \
    APP_ENV=production \
    PORT=5000

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:5000/health || exit 1

# Seed demo data on first boot if the database doesn't exist yet, then start
# the production WSGI server. See docker-entrypoint.sh.
ENTRYPOINT ["./docker-entrypoint.sh"]
