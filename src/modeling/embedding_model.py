import os
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from transformers import BertTokenizer, BertModel
import tensorflow_hub as hub
import gensim.downloader as api
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV
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
            if self.model_name == 'word2vec':
                embeddings = []
                for text in tqdm(texts, desc="Generating Word2Vec embeddings"):
                    words = text.lower().split()
                    vectors = [self.embedding_model[word]
                               for word in words if word in self.embedding_model]
                    embeddings.append(np.mean(vectors, axis=0) if vectors else np.zeros(
                        self.embedding_model.vector_size))
                return np.array(embeddings)

            elif self.model_name == 'bert':
                embeddings = []
                for text in tqdm(texts, desc="Generating BERT embeddings"):
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
        # Generate timestamp for unique model names
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_run_name = f"{self.model_name}_{timestamp}"

        # Log data information
        logger.info(f"\nData Information:")
        logger.info(f"Training set size: {X_train.shape}")
        logger.info(f"Test set size: {X_test.shape}")
        logger.info(f"Number of features: {X_train.shape[1]}")
        logger.info(f"Number of labels: {y_train.shape[1]}")

        # Create directory for reports
        reports_dir = 'outputs/reports'
        os.makedirs(reports_dir, exist_ok=True)

        # Get classifier parameters
        classifier_name = config['embedding']['classifier']
        classifier_config = config['embedding']['classifiers'].get(
            classifier_name, {})
        base_params = classifier_config.get('base_params', {})
        search_params = classifier_config.get('search_params', None)

        # Create and train classifier
        base_model = self.get_base_classifier(classifier_name, base_params)
        multi_model = MultiOutputClassifier(base_model)

        if search_params:
            random_search = RandomizedSearchCV(
                estimator=multi_model,
                param_distributions=search_params,
                n_iter=5,
                cv=3,
                scoring='f1_micro',
                verbose=1,
                n_jobs=-1,
                random_state=42
            )

            logger.info(f"Launch RandomizedSearchCV for {classifier_name}...")
            random_search.fit(X_train, y_train)
            self.classifier = random_search.best_estimator_

            # Save search results
            cv_results = pd.DataFrame(random_search.cv_results_)
            cv_results_path = os.path.join(
                reports_dir, f"cv_results_{model_run_name}.csv")
            cv_results.to_csv(cv_results_path, index=False)
        else:
            logger.info(f"Training {classifier_name} with base parameters...")
            self.classifier = multi_model
            self.classifier.fit(X_train, y_train)

        # Evaluate model
        predictions = self.classifier.predict(X_test)
        f1 = f1_score(y_test, predictions, average='micro')
        precision = precision_score(y_test, predictions, average='micro')
        recall = recall_score(y_test, predictions, average='micro')

        logger.info(f"\nModel performance metrics:")
        logger.info(f"F1 micro: {f1:.4f}")
        logger.info(f"Precision micro: {precision:.4f}")
        logger.info(f"Recall micro: {recall:.4f}")

        # Save detailed report
        report = classification_report(y_test, predictions)
        logger.info("\nClassification Report:")
        logger.info(report)

        report_path = os.path.join(
            reports_dir, f"classification_report_{model_run_name}.txt")
        with open(report_path, "w") as f:
            f.write(report)

        # Save model with timestamp
        model_dir = 'models/embedding'
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{model_run_name}.pkl")
        joblib.dump(self.classifier, model_path)

        return self.classifier, {
            'model_path': model_path,
            'report_path': report_path,
            'cv_results_path': cv_results_path if search_params else None,
            'metrics': {
                'f1_micro': f1,
                'precision_micro': precision,
                'recall_micro': recall
            },
            'predictions': predictions,
            'y_test': y_test,
            'model_name': self.model_name,
            'timestamp': timestamp,
            'search_params': search_params,
            'best_params': random_search.best_params_ if search_params else None,
            'best_score': random_search.best_score_ if search_params else None,
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
