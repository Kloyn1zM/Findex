# syntax=docker/dockerfile:1

# --- Stage 1: Builder (installs locked base dependencies without dev or semantic extras) ---
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Install third-party dependencies first for optimal layer caching
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy package sources (including Jinja2 templates) and install findex into .venv
COPY src ./src
COPY README.md ./
RUN uv sync --frozen --no-dev --no-editable


# --- Stage 2: Minimal non-root runtime image ---
FROM python:3.13-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    HOST="0.0.0.0" \
    PORT="8000" \
    WORKERS="1"

WORKDIR /app

# Create unprivileged user before copying application files
RUN groupadd --system --gid 1001 appgroup \
    && useradd --system --uid 1001 --gid appgroup --home /app --shell /usr/sbin/nologin appuser

COPY --from=builder --chown=appuser:appgroup /app/.venv /app/.venv

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8000\")}/health', timeout=3)"]

# Expects INDEX_PATH (and optionally PORT / WORKERS) via environment variables
CMD ["findex", "serve"]
