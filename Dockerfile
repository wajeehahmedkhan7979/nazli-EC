FROM python:3.11-slim

# System deps: poppler (PDF), tesseract + Japanese language pack, libgl (OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    tesseract-ocr \
    tesseract-ocr-jpn \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"

COPY . .

ENV THRESHOLDS_PATH=configs/thresholds.yaml
ENV REGISTRY_PATH=configs/templates/template_registry.yaml
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["python", "main.py"]
