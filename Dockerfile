# Base image for IteraTex services
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1\
    PIP_NO_CACHE_DIR=1

# Install system deps for pyarrow
RUN apt-get update -y && apt-get install -y --no-install-recommends \
        gcc g++ build-essential curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY iteratex /app/iteratex

# Install project in editable mode
RUN pip install --upgrade pip && \
    pip install .

CMD ["uvicorn", "iteratex.serving.main:app", "--host", "0.0.0.0", "--port", "8000"]
