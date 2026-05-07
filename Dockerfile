FROM python:3.11-slim

# Install Tesseract with Japanese + English language packs
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-jpn \
    tesseract-ocr-eng \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

COPY . .

RUN mkdir -p data/results

ENV PYTHONUNBUFFERED=1
ENV RESULTS_DIR=/app/data/results

EXPOSE 8080
CMD ["python", "main.py"]
