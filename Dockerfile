FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY cracks/ cracks/
COPY utils/ utils/
COPY main.py .

ENTRYPOINT ["python", "/app/main.py"]