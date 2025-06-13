FROM python:3.10.16-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt || \
    pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2 -r requirements.txt

COPY . .

RUN mkdir -p backend/model && \
    wget --tries=3 -O backend/model/best_glaucoma_model.pth \
    https://huggingface.co/5t4l1n/ai-eye-disease-detection/resolve/main/model/best_glaucoma_model.pth || \
    { echo "Model download failed"; exit 1; }

EXPOSE 5000 8000

ENV FLASK_ENV=development
ENV FLASK_DEBUG=True
ENV FLASK_HOST=0.0.0.0
ENV FLASK_PORT=5000
ENV MODEL_PATH=/app/backend/model/best_glaucoma_model.pth
ENV MAX_CONTENT_LENGTH=16777216
ENV ALLOWED_ORIGINS=http://localhost:8000

CMD ["python", "server.py"]
