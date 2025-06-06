import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.pyfunc
import numpy as np
import pandas as pd
from typing import Dict, Any
import os
from datetime import datetime
import scipy.sparse
import logging

logger = logging.getLogger(__name__)


class MLflowRecipe:
    def __init__(self, config: Dict[str, Any]):
        """Initialize MLflow recipe"""
        self.config = config
        self.task_type = self.config['task_type']
        self.model_name = self.config[self.task_type]['model']

        # Initialize embedding model if needed
        if self.task_type == 'embedding':
            from src.modeling.embedding_model import EmbeddingModel
            self.embedding_model = EmbeddingModel(self.model_name)

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log metrics to MLflow and save to files"""
        try:
            # Log overall metrics to MLflow
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)

            # Create metrics directory if it doesn't exist
            metrics_dir = 'outputs/metrics'
            os.makedirs(metrics_dir, exist_ok=True)

            # Get current model name from MLflow tags
            model_name = mlflow.get_run(mlflow.active_run().info.run_id).data.tags.get(
                'model_name', 'unknown_model')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Save overall metrics to a single file
            metrics_file = os.path.join(
                metrics_dir, f'{model_name}_metrics_{timestamp}.txt')
            with open(metrics_file, 'w') as f:
                f.write(f"Model: {model_name}\n")
                f.write(f"Timestamp: {timestamp}\n")
                for metric_name, value in metrics.items():
                    f.write(f"{metric_name}: {value:.4f}\n")

        except Exception as e:
            logger.error(f"Error logging metrics: {str(e)}")
            raise

    def log_data_stats(self, X_train, y_train, X_test, y_test):
        """Log data statistics to MLflow"""
        try:
            # Convert inputs to numpy arrays if they are lists
            X_train = np.array(X_train) if isinstance(
                X_train, list) else X_train
            X_test = np.array(X_test) if isinstance(X_test, list) else X_test
            y_train = np.array(y_train) if isinstance(
                y_train, list) else y_train
            y_test = np.array(y_test) if isinstance(y_test, list) else y_test

            # Handle different input formats
            if isinstance(X_train, pd.DataFrame):
                n_features = X_train.shape[1]
                n_samples = len(X_train)
            elif isinstance(X_train, np.ndarray):
                if len(X_train.shape) == 1:
                    n_features = 1
                    n_samples = len(X_train)
                else:
                    n_features = X_train.shape[1]
                    n_samples = X_train.shape[0]
            elif scipy.sparse.issparse(X_train):
                n_features = X_train.shape[1]
                n_samples = X_train.shape[0]
            else:
                n_features = X_train.shape[1] if hasattr(
                    X_train, 'shape') else len(X_train[0])
                n_samples = X_train.shape[0] if hasattr(
                    X_train, 'shape') else len(X_train)

            # Log basic statistics
            mlflow.log_metric("n_samples", n_samples)
            mlflow.log_metric("n_features", n_features)
            mlflow.log_metric("n_classes", y_train.shape[1] if len(
                y_train.shape) > 1 else 1)

            # Log test set size
            if scipy.sparse.issparse(X_test):
                test_size = X_test.shape[0]
            else:
                test_size = len(X_test)
            mlflow.log_metric("test_set_size", test_size)

        except Exception as e:
            logger.error(f"Error logging data stats: {str(e)}")
            raise

    def log_config(self) -> None:
        """Configuration logs"""
        mlflow.log_params(self.config[self.task_type])
        if self.task_type == 'supervised':
            mlflow.log_params(self.config['vectorizer'])

    def log_artifacts(self, vectorizer_path: str = None) -> None:
        """Artifact logs"""
        if vectorizer_path:
            mlflow.log_artifact(vectorizer_path, "vectorizer")
        mlflow.log_artifact("config/config.yaml", "config")

    def _log_model_artifacts(self, model_results: Dict[str, Any]) -> None:
        """Log all artifacts from model results"""
        artifact_paths = {
            'report_path': None,
            'cv_results_path': None,
            'model_path': None,
            'visualization_path': None,
            'distribution_path': None
        }

        for path_key in artifact_paths:
            if path_key in model_results and model_results[path_key]:
                mlflow.log_artifact(model_results[path_key])

    def _log_model_metrics(self, model_results: Dict[str, Any]) -> None:
        """Log all metrics from model results"""
        if 'metrics' in model_results:
            for metric_name, value in model_results['metrics'].items():
                mlflow.log_metric(metric_name, value)

            if 'predictions' in model_results and 'y_test' in model_results:
                self.log_metrics(model_results['metrics'])

    def _log_search_params(self, model_results: Dict[str, Any]) -> None:
        """Log search parameters if they exist"""
        if model_results.get('search_params'):
            mlflow.set_tag("tuning", "RandomizedSearchCV")
            for param, values in model_results['search_params'].items():
                mlflow.log_param(f"search_{param}", values)
            mlflow.log_params(model_results['best_params'])
            mlflow.log_metric("best_cv_score", model_results['best_score'])

    def _log_model(self, model_results):
        """Log model to MLflow"""
        try:
            # Get model and metadata
            model = model_results['model']
            model_name = model_results['model_name']
            timestamp = model_results['timestamp']
            metrics = model_results['metrics']
            predictions = model_results['predictions']
            y_test = model_results['y_test']

            # Log model
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name=f"{model_name}_{timestamp}"
            )

            # Log metrics
            self.log_metrics(metrics)

            # Log predictions and test data
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.npy') as tmp:
                np.save(tmp.name, predictions)
                mlflow.log_artifact(tmp.name, "predictions.npy")

            with tempfile.NamedTemporaryFile(suffix='.npy') as tmp:
                np.save(tmp.name, y_test)
                mlflow.log_artifact(tmp.name, "y_test.npy")

            # Log model metadata
            mlflow.set_tag("model_name", model_name)
            mlflow.set_tag("timestamp", timestamp)
            if self.task_type == 'embedding':
                mlflow.set_tag("embedding_model",
                               model_results['embedding_model'])
                mlflow.set_tag("classifier_name",
                               model_results['classifier_name'])

        except Exception as e:
            logger.error(f"Error logging model: {str(e)}")
            raise

    def log_model_results(self, model_results: Dict[str, Any]) -> None:
        """Logs model learning results"""
        try:
            # Set task-specific tags first
            self._set_task_tags(model_results)

            # Log task-specific parameters
            if self.task_type == 'supervised':
                model_config = self.config['supervised']['models'].get(
                    self.model_name, {})
                if 'base_params' in model_config:
                    mlflow.log_params(model_config['base_params'])
            elif self.task_type == 'lda':
                mlflow.log_params({
                    'num_topics': self.config['lda']['num_topics'],
                    'passes': self.config['lda']['passes']
                })
            elif self.task_type == 'embedding':
                embedding_config = self.config['embedding']['models'].get(
                    self.model_name, {})
                if 'base_params' in embedding_config:
                    mlflow.log_params(embedding_config['base_params'])
                classifier_name = self.config['embedding']['classifier']
                classifier_config = self.config['embedding']['classifiers'].get(
                    classifier_name, {})
                if 'base_params' in classifier_config:
                    mlflow.log_params(classifier_config['base_params'])

            # Log common artifacts and metrics
            self._log_model_artifacts(model_results)
            self._log_model_metrics(model_results)
            self._log_search_params(model_results)

            # Log the model
            self._log_model(model_results)

        except Exception as e:
            logger.error(f"Error logging model results: {str(e)}")
            raise

    def _set_task_tags(self, model_results: Dict[str, Any]) -> None:
        """Set task-specific tags"""
        task_tags = {
            'supervised': "multi-label classification",
            'embedding': "embedding-based classification",
            'lda': "topic modeling"
        }

        # Set model name first
        mlflow.set_tag("model_name", model_results['model_name'])

        # Set other tags
        mlflow.set_tag("task", task_tags.get(self.task_type))
        mlflow.set_tag(
            "run_name", f"{model_results['model_name']}_{model_results['timestamp']}")
        mlflow.set_tag(
            "use_sample", self.config['data'].get('use_sample', False))

        if self.task_type == 'embedding':
            mlflow.log_param("embedding_model",
                             model_results['embedding_model'])
            mlflow.log_param("classifier", model_results['classifier_name'])


def create_recipe(config: Dict[str, Any]) -> MLflowRecipe:
    """Create MLflow recipe according to the type of task"""
    return MLflowRecipe(config)
