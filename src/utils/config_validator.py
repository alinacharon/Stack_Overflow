import yaml
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def validate_config(config):
    """Validate configuration dictionary"""
    required_sections = ['task_type', 'data', 'mlflow', 'vectorizer']

    # Check required sections
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section: {section}")

    # Validate task type
    valid_task_types = ['supervised', 'lda', 'embedding']
    if config['task_type'] not in valid_task_types:
        raise ValueError(
            f"Invalid task type. Must be one of: {valid_task_types}")

    # Validate data section
    data_section = config['data']
    if not isinstance(data_section, dict):
        raise ValueError("Data section must be a dictionary")

    # Validate vectorizer section
    vectorizer_section = config['vectorizer']
    if not isinstance(vectorizer_section, dict):
        raise ValueError("Vectorizer section must be a dictionary")
    if 'method' not in vectorizer_section:
        raise ValueError("Vectorizer method not specified")
    if vectorizer_section['method'] not in ['tfidf', 'count']:
        raise ValueError(
            "Invalid vectorizer method. Must be 'tfidf' or 'count'")

    # Validate task-specific sections
    task_type = config['task_type']
    if task_type == 'supervised':
        if 'supervised' not in config:
            raise ValueError("Missing supervised section")
        if 'model' not in config['supervised']:
            raise ValueError("Model not specified in supervised section")
    elif task_type == 'lda':
        if 'lda' not in config:
            raise ValueError("Missing LDA section")
    elif task_type == 'embedding':
        if 'embedding' not in config:
            raise ValueError("Missing embedding section")
        if 'model' not in config['embedding']:
            raise ValueError("Model not specified in embedding section")

    # Validate data paths
    required_data_paths = ['train_path', 'target_path']
    for path in required_data_paths:
        if not Path(data_section[path]).exists():
            raise ValueError(
                f"Data path does not exist: {data_section[path]}")

    # Validate supervised model if task type is supervised
    if task_type == 'supervised':
        valid_models = ['logreg', 'rf', 'xgboost']
        model_name = config['supervised']['model']
        if model_name not in valid_models:
            raise ValueError(f"Unsupported model: {model_name}")

    # Validate embedding model if task type is embedding
    if task_type == 'embedding':
        valid_embedding_models = ['word2vec', 'bert', 'use']
        embedding_model = config['embedding']['model']
        if embedding_model not in valid_embedding_models:
            raise ValueError(f"Unsupported embedding model: {embedding_model}")

        valid_classifiers = ['logreg', 'rf', 'xgboost']
        classifier_name = config['embedding']['classifier']
        if classifier_name not in valid_classifiers:
            raise ValueError(
                f"Unsupported classifier for embedding: {classifier_name}")

    # Validate LDA parameters if task type is lda
    if task_type == 'lda':
        if 'model' not in config['lda']:
            raise ValueError("Model not specified in LDA section")
        if config['lda']['model'] != 'lda':
            raise ValueError("Invalid LDA model name")
        if not isinstance(config['lda']['num_topics'], int) or config['lda']['num_topics'] <= 0:
            raise ValueError("num_topics must be a positive integer")
        if not isinstance(config['lda']['passes'], int) or config['lda']['passes'] <= 0:
            raise ValueError("passes must be a positive integer")
        if 'visualization' not in config['lda']:
            raise ValueError("Missing visualization section in LDA config")
        required_vis_paths = ['output_dir',
                              'topics_vis_path', 'distribution_vis_path']
        for path in required_vis_paths:
            if path not in config['lda']['visualization']:
                raise ValueError(
                    f"Missing required visualization path: {path}")

    return True
