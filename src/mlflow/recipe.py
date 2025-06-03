import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.pyfunc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
from typing import Dict, Any


class MLflowRecipe:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.task_type = config['task_type']
        self.model_name = config[self.task_type]['model']

    def log_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Metric logs for Multipoint Classification"""
        # For each tag calculate metrics
        for i in range(y_true.shape[1]):
            tag_name = f"tag_{i}"
            mlflow.log_metric(f"{tag_name}_accuracy",
                              accuracy_score(y_true[:, i], y_pred[:, i]))
            mlflow.log_metric(f"{tag_name}_precision", precision_score(
                y_true[:, i], y_pred[:, i], zero_division=0))
            mlflow.log_metric(f"{tag_name}_recall", recall_score(
                y_true[:, i], y_pred[:, i], zero_division=0))
            mlflow.log_metric(f"{tag_name}_f1", f1_score(
                y_true[:, i], y_pred[:, i], zero_division=0))

    def log_model_params(self, model: Any) -> None:
        """Model Parameter logs"""
        if self.task_type == 'supervised':
            if self.model_name == 'xgboost':
                mlflow.xgboost.log_model(model, "model")
            else:
                mlflow.sklearn.log_model(model, "model")
        elif self.task_type == 'lda':
            mlflow.pyfunc.log_model(model, "model")
        elif self.task_type == 'embedding':
            mlflow.pyfunc.log_model(model, "model")

    def log_data_stats(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
        """Data statistics log"""
        mlflow.log_metric("n_samples", X_train.shape[0])
        mlflow.log_metric("n_features", X_train.shape[1])
        mlflow.log_metric("n_tags", y_train.shape[1])

        avg_tags = y_train.sum(axis=1).mean()
        mlflow.log_metric("avg_tags_per_sample", avg_tags)

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

    def log_model_results(self, model_results: Dict[str, Any]) -> None:
        """Logs model learning results"""
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
            if 'metrics' in model_results:
                for metric_name, value in model_results['metrics'].items():
                    mlflow.log_metric(metric_name, value)
            if 'visualization_path' in model_results:
                mlflow.log_artifact(model_results['visualization_path'])
            if 'distribution_path' in model_results:
                mlflow.log_artifact(model_results['distribution_path'])
            mlflow.pyfunc.log_model(
                model_results['model'],
                artifact_path=f"models/lda/lda_model_{model_results.get('timestamp', '')}",
                registered_model_name="lda_model"
            )
            mlflow.set_tag("task", "topic modeling")
            mlflow.set_tag("model_name", "lda")
            mlflow.set_tag(
                "run_name", f"lda_{model_results.get('timestamp', '')}")
            mlflow.set_tag(
                "use_sample", self.config['data'].get('use_sample', False))
            return
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

        if 'metrics' in model_results:
            for metric_name, value in model_results['metrics'].items():
                mlflow.log_metric(metric_name, value)

            if self.task_type == 'supervised' and 'predictions' in model_results and 'y_test' in model_results:
                self.log_metrics(
                    model_results['y_test'], model_results['predictions'])

        if model_results.get('search_params'):
            mlflow.set_tag("tuning", "RandomizedSearchCV")
            for param, values in model_results['search_params'].items():
                mlflow.log_param(f"search_{param}", values)
            mlflow.log_params(model_results['best_params'])
            mlflow.log_metric("best_cv_score", model_results['best_score'])

        if 'report_path' in model_results:
            mlflow.log_artifact(model_results['report_path'])
        if model_results.get('cv_results_path'):
            mlflow.log_artifact(model_results['cv_results_path'])
        if 'model_path' in model_results:
            mlflow.log_artifact(model_results['model_path'])
        if 'visualization_path' in model_results:
            mlflow.log_artifact(model_results['visualization_path'])
        if 'distribution_path' in model_results:
            mlflow.log_artifact(model_results['distribution_path'])

        if self.task_type == 'supervised':
            if self.model_name == 'xgboost':
                mlflow.xgboost.log_model(
                    model_results['model'],
                    artifact_path=f"models/supervised/{model_results['model_name']}_{model_results['timestamp']}",
                    registered_model_name=f"supervised_{model_results['model_name']}"
                )
            else:
                mlflow.sklearn.log_model(
                    model_results['model'],
                    artifact_path=f"models/supervised/{model_results['model_name']}_{model_results['timestamp']}",
                    registered_model_name=f"supervised_{model_results['model_name']}"
                )
        elif self.task_type == 'embedding':
            mlflow.pyfunc.log_model(
                model_results['model'],
                artifact_path=f"models/embedding/{model_results['model_name']}_{model_results['timestamp']}",
                registered_model_name=f"embedding_{model_results['embedding_model']}"
            )

        if self.task_type == 'supervised':
            mlflow.set_tag("task", "multi-label classification")
        elif self.task_type == 'embedding':
            mlflow.set_tag("task", "embedding-based classification")
            mlflow.log_param("embedding_model",
                             model_results['embedding_model'])
            mlflow.log_param("classifier", model_results['classifier_name'])

        mlflow.set_tag("model_name", model_results['model_name'])
        mlflow.set_tag(
            "run_name", f"{model_results['model_name']}_{model_results['timestamp']}")
        mlflow.set_tag(
            "use_sample", self.config['data'].get('use_sample', False))


def create_recipe(config: Dict[str, Any]) -> MLflowRecipe:
    """Create MLflow recipe according to the type of task"""
    return MLflowRecipe(config)
