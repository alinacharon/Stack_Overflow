from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import logging
import os
from typing import List
import time
from contextlib import asynccontextmanager
from .model_loader import load_model_and_vectorizer
import re
import numpy as np

# Configuration logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Port
PORT = int(os.getenv('API_PORT', '3000'))

# Global variables
global model, embedding_model
model = None
embedding_model = None


def preprocess_text(text: str, model_type: str = 'supervised') -> str:
    """Preprocess text for prediction"""
    if model_type == 'supervised':
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
    elif model_type in ['bert', 'use']:
        # For BERT and USE, just clean whitespace
        text = ' '.join(text.split())
    logger.info(f"Preprocessed text: '{text}'")
    return text


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app"""
    # Load model and embedding model on startup
    global model, embedding_model
    model, embedding_model = load_model_and_vectorizer()
    yield
    # Cleanup on shutdown
    model = None
    embedding_model = None

app = FastAPI(
    title="Stack Overflow Tag Predictor",
    description="API to predict tags for Stack Overflow questions",
    version="1.0.0",
    lifespan=lifespan
)


class Question(BaseModel):
    text: str = Field(..., min_length=1,
                      description="Question text to predict tags for")


class Prediction(BaseModel):
    tags: List[str] = Field(..., min_length=1, description="Predicted tags")
    probabilities: List[float] = Field(..., min_length=1,
                                       description="Probabilities for each tag")


@app.post("/predict", response_model=Prediction)
async def predict(question: Question):
    try:
        global model, embedding_model
        if model is None or embedding_model is None:
            # Try to load model if not loaded
            model, embedding_model = load_model_and_vectorizer()
            if model is None or embedding_model is None:
                raise HTTPException(status_code=500, detail="Model not loaded")

        start_time = time.time()
        logger.info(f"Original text: '{question.text}'")

        # Get model type
        model_type = os.getenv('MODEL_TYPE', 'supervised')

        # Preprocess text
        processed_text = preprocess_text(question.text, model_type)
        if not processed_text:
            raise HTTPException(
                status_code=400, detail="Invalid input text after preprocessing")

        # Get embeddings or transform text
        if model_type in ['bert', 'use']:
            X = embedding_model.get_embeddings([processed_text])
        else:
            X = embedding_model.transform([processed_text])
        logger.info(f"Vector shape: {X.shape}")

        # Get predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        logger.info(f"Predictions: {predictions}")
        logger.info(f"Probabilities: {probabilities}")

        # Get tags and probabilities
        tags = []
        probs = []

        # Handle both list and array predictions
        if isinstance(predictions, list):
            preds = predictions[0] if len(predictions) > 0 else []
            probs_list = probabilities[0] if len(probabilities) > 0 else []
        else:
            preds = predictions[0]
            probs_list = probabilities[0]

        for i, (pred, prob) in enumerate(zip(preds, probs_list)):
            if pred == 1:
                tags.append(model.classes_[i])
                probs.append(float(prob))

        logger.info(f"Found tags: {tags}")
        if not tags:
            raise HTTPException(status_code=400, detail="No tags predicted")

        processing_time = time.time() - start_time
        logger.info(f"Prediction fulfilled in {processing_time:.2f} seconds")
        logger.info(f"Text: {question.text[:100]}...")
        logger.info(f"Predicted tags: {tags}")

        return Prediction(tags=tags, probabilities=probs)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    logger.info("API health check")
    return {"status": "healthy"}
