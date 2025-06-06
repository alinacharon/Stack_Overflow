import numpy as np
import logging
from datetime import datetime
from transformers import BertTokenizer, BertModel
import tensorflow_hub as hub
import gensim.downloader as api
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import joblib
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


class EmbeddingModel:
    def __init__(self, model_name, model_params=None):
        self.model_name = model_name
        self.model_params = model_params or {}
        self.embedding_model = None
        self.tokenizer = None
        self.classifier = None
        self.initialize_model()

    def initialize_model(self):
        """Initialize embedding model"""
        try:
            if self.model_name == 'word2vec':
                logger.info("Loading Word2Vec model...")
                self.embedding_model = api.load('word2vec-google-news-300')
                logger.info("Word2Vec model loaded successfully")

            elif self.model_name == 'bert':
                logger.info("Loading BERT model...")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'bert-base-uncased')
                self.embedding_model = BertModel.from_pretrained(
                    'bert-base-uncased')
                self.embedding_model.eval()
                logger.info("BERT model loaded successfully")

            elif self.model_name == 'use':
                logger.info("Loading USE model...")
                self.embedding_model = hub.load(
                    "https://tfhub.dev/google/universal-sentence-encoder/4")
                logger.info("USE model loaded successfully")

            else:
                raise ValueError(f"Unknown model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise

    def get_embeddings(self, texts):
        """Get embeddings for texts"""
        try:
            if not isinstance(texts, (list, np.ndarray)):
                texts = [texts]

            if self.model_name == 'word2vec':
                embeddings = []
                for text in tqdm(texts, desc="Generating Word2Vec embeddings"):
                    if not isinstance(text, str):
                        text = str(text)
                    words = text.lower().split()
                    vectors = [self.embedding_model[word]
                               for word in words if word in self.embedding_model]
                    if not vectors:
                        logger.warning(
                            f"No vectors found for text: {text[:100]}...")
                        vectors = [np.zeros(self.embedding_model.vector_size)]
                    embeddings.append(np.mean(vectors, axis=0))
                return np.array(embeddings)

            elif self.model_name == 'bert':
                embeddings = []
                for text in tqdm(texts, desc="Generating BERT embeddings"):
                    if not isinstance(text, str):
                        text = str(text)
                    inputs = self.tokenizer(
                        text, return_tensors='pt', truncation=True, padding=True, max_length=512)
                    with torch.no_grad():
                        outputs = self.embedding_model(**inputs)
                    embeddings.append(
                        outputs.last_hidden_state[:, 0, :].squeeze().numpy())
                return np.array(embeddings)

            elif self.model_name == 'use':
                embeddings = []
                batch_size = 32
                for i in tqdm(range(0, len(texts), batch_size), desc="Generating USE embeddings"):
                    batch = texts[i:i + batch_size]
                    # Convert all elements to strings
                    batch = [str(text) for text in batch]
                    batch_embeddings = self.embedding_model(batch).numpy()
                    embeddings.append(batch_embeddings)
                return np.vstack(embeddings)

        except Exception as e:
            logger.error(f"Error getting embeddings: {str(e)}")
            raise

    def get_base_classifier(self, classifier_name, classifier_params):
        """Get base classifier"""
        if classifier_name == 'logreg':
            return LogisticRegression(**classifier_params)
        elif classifier_name == 'rf':
            return RandomForestClassifier(**classifier_params)
        elif classifier_name == 'xgboost':
            return XGBClassifier(**classifier_params)
        else:
            raise ValueError(f"Unknown classifier: {classifier_name}")

    def train_classifier(self, X_train, y_train, X_test, y_test, config):
        """Train classifier on embeddings"""
        # Convert inputs to numpy arrays if they are lists
        X_train = np.array(X_train) if isinstance(X_train, list) else X_train
        X_test = np.array(X_test) if isinstance(X_test, list) else X_test
        y_train = np.array(y_train) if isinstance(y_train, list) else y_train
        y_test = np.array(y_test) if isinstance(y_test, list) else y_test

        # Generate timestamp for unique model names
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_run_name = f"{self.model_name}_{timestamp}"

        # Get embeddings
        X_train_embeddings = self.get_embeddings(X_train)
        X_test_embeddings = self.get_embeddings(X_test)

        # Initialize classifier
        classifier_name = config['embedding']['classifier']
        classifier_config = config['embedding']['classifiers'].get(
            classifier_name, {})
        base_params = classifier_config.get('base_params', {})
        search_params = classifier_config.get('search_params', {})

        if classifier_name == 'logistic':
            from sklearn.linear_model import LogisticRegression
            classifier = LogisticRegression(**base_params)
        elif classifier_name == 'svm':
            from sklearn.svm import SVC
            classifier = SVC(**base_params)
        elif classifier_name == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            classifier = RandomForestClassifier(**base_params)
        elif classifier_name == 'xgboost':
            from xgboost import XGBClassifier
            classifier = XGBClassifier(**base_params)
        else:
            raise ValueError(f"Unsupported classifier: {classifier_name}")

        # Train classifier
        classifier.fit(X_train_embeddings, y_train)

        # Make predictions
        y_pred = classifier.predict(X_test_embeddings)

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_micro': precision_score(y_test, y_pred, average='micro', zero_division=0),
            'recall_micro': recall_score(y_test, y_pred, average='micro', zero_division=0),
            'f1_micro': f1_score(y_test, y_pred, average='micro', zero_division=0)
        }

        # Save model
        model_path = f"models/embedding/{model_run_name}.pkl"
        joblib.dump(classifier, model_path)

        return classifier, {
            'model': classifier,
            'model_name': model_run_name,
            'timestamp': timestamp,
            'metrics': metrics,
            'predictions': y_pred,
            'y_test': y_test,
            'embedding_model': self.model_name,
            'classifier_name': classifier_name
        }

    def save_model(self, path):
        """Save model"""
        joblib.dump(self, path)

    @classmethod
    def load_model(cls, path):
        """Load model"""
        return joblib.load(path)
