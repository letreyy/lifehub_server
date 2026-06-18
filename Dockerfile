FROM python:3.10-slim

# Use stable and fast mirror for apt-get (supports both legacy and DEB822 formats)
RUN if [ -f /etc/apt/sources.list ]; then \
        sed -i 's/deb.debian.org/mirror.yandex.ru/g' /etc/apt/sources.list && \
        sed -i 's/security.debian.org/mirror.yandex.ru/g' /etc/apt/sources.list; \
    fi && \
    if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
        sed -i 's/deb.debian.org/mirror.yandex.ru/g' /etc/apt/sources.list.d/debian.sources && \
        sed -i 's/security.debian.org/mirror.yandex.ru/g' /etc/apt/sources.list.d/debian.sources; \
    fi

# Install system dependencies required by OpenCV and PaddleOCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-create directory for PaddleOCR model cache
RUN mkdir -p /root/.paddleocr

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

ENV OLLAMA_URL=http://host.docker.internal:11434
ENV OLLAMA_MODEL=qwen2.5:3b

CMD ["python", "main.py"]
