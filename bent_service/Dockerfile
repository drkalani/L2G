FROM python:3.10.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/service \
    PIP_NO_CACHE_DIR=1

WORKDIR /service

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       gcc \
    && rm -rf /var/lib/apt/lists/*

COPY l2g_core /service/l2g_core
COPY bent_service /service/bent_service

RUN pip install --upgrade pip \
    && pip install bent==0.0.80

EXPOSE 8010

CMD ["python", "-m", "bent_service.main", "--host", "0.0.0.0", "--port", "8010"]
