FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app


COPY requirements.txt .


RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY models/final/ ./models/final/


RUN mkdir -p logs

ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/models/final/model.pkl
ENV VECTORIZER_PATH=/app/models/final/vectorizer.pkl
ENV API_HOST=0.0.0.0
ENV API_PORT=3000


EXPOSE 3000


CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "3000"] 