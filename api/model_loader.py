import os
import joblib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_model_and_vectorizer():
    """Load model and vectorizer"""
    try:
        # Get project root directory
        project_root = str(Path(__file__).parent.parent)

        # Get model paths from environment or use defaults
        model_path = os.getenv('MODEL_PATH', os.path.join(
            project_root, 'models/final/model.pkl'))
        vectorizer_path = os.getenv('VECTORIZER_PATH', os.path.join(
            project_root, 'models/final/vectorizer.pkl'))

        logger.info(f"Looking for model at: {model_path}")
        logger.info(f"Looking for vectorizer at: {vectorizer_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found on: {model_path}")
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Vector not found on: {vectorizer_path}")

        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)

        logger.info(f"Model successfully loaded from {model_path}")
        logger.info(f"Vector successfully loaded from {vectorizer_path}")
        logger.info(f"Number of model classes: {len(model.classes_)}")

        return model, vectorizer
    except Exception as e:
        logger.error(f"Model loading error: {str(e)}")
        return None, None
