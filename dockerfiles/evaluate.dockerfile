# Base image
FROM python:3.10-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY src src/
COPY data data/
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml

WORKDIR /

RUN --mount=type=cache,target=~/pip/.cache pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir --verbose

ENTRYPOINT ["python", "-u", "src/cmnist_cnn/evaluate.py"]