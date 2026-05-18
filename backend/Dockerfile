FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/backend:/app \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/pyproject.toml
COPY l2g_core /app/l2g_core
COPY backend /app/backend

RUN pip install --upgrade pip \
    && PIP_INDEX_URL=https://download.pytorch.org/whl/cpu \
       PIP_EXTRA_INDEX_URL=https://pypi.org/simple \
       pip install -e ".[api]"

WORKDIR /app/backend

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
