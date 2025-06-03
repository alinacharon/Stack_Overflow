import joblib
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV
import scipy.sparse
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)


def get_base_model(model_name, model_params):
    """Get base model instance with specified parameters"""
    if model_name == 'logreg':
        return LogisticRegression(**model_params)
    elif model_name == 'rf':
        return RandomForestClassifier(**model_params)
    elif model_name == 'xgboost':
        return XGBClassifier(**model_params)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def train_classifier(X_train, y_train, X_test, y_test, model_name='logreg', config=None):
    """Train classifier with specified model and parameters"""
    # Generate timestamp for unique model names
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_run_name = f"{model_name}_{timestamp}"

    # Log data information
    logger.info(f"\nData Information:")
    logger.info(f"Training set size: {X_train.shape}")
    logger.info(f"Test set size: {X_test.shape}")
    logger.info(f"Number of features: {X_train.shape[1]}")
    logger.info(f"Number of labels: {y_train.shape[1]}")
    logger.info(
        f"Using sample data: {config['data'].get('use_sample', False)}")
    if config['data'].get('use_sample', False):
        logger.info(f"Sample size: {config['data']['sample']['size']}")
        logger.info(f"Top N tags: {config['data']['sample']['top_n_tags']}")

    # Check that each label has at least 2 classes
    columns_to_keep = []
    for col in y_train.columns:
        unique_classes = y_train[col].unique()
        if len(unique_classes) < 2:
            logger.warning(
                f"Label {col} has only one class: {unique_classes[0]}. Removing this label.")
        else:
            columns_to_keep.append(col)

    y_train = y_train[columns_to_keep]
    y_test = y_test[columns_to_keep]
    logger.info(f"Number of labels after cleaning: {len(columns_to_keep)}")

    # Create directory for reports
    reports_dir = 'outputs/reports'
    os.makedirs(reports_dir, exist_ok=True)

    model_config = config['supervised']['models'].get(model_name, {})
    base_params = model_config.get('base_params', {})
    search_params = model_config.get('search_params', None)

    base_model = get_base_model(model_name, base_params)
    multi_model = MultiOutputClassifier(base_model)

    def safe_cast(X):
        """Safely cast input data to float32"""
        if scipy.sparse.issparse(X):
            return X.astype(np.float32)
        elif isinstance(X, pd.DataFrame):
            return X.values.astype(np.float32)
        elif isinstance(X, np.ndarray):
            return X.astype(np.float32)
        else:
            raise TypeError(f"Unsupported input type: {type(X)}")

    def safe_cast_y(y):
        """Safely cast target data to int32"""
        if isinstance(y, pd.DataFrame):
            return y.values.astype(np.int32)
        elif isinstance(y, np.ndarray):
            return y.astype(np.int32)
        else:
            return np.array(y, dtype=np.int32)

    # Prepare data
    X_train = safe_cast(X_train)
    X_test = safe_cast(X_test)
    y_train = safe_cast_y(y_train)
    y_test = safe_cast_y(y_test)

    if search_params:
        logger.info(f"Starting RandomizedSearchCV for {model_name}...")
        n_iter = 2
        random_search = RandomizedSearchCV(
            estimator=multi_model,
            param_distributions=search_params,
            n_iter=n_iter,
            cv=2,
            scoring='f1_micro',
            verbose=1,
            n_jobs=-1,
            random_state=42
        )
        random_search.fit(X_train, y_train)
        model = random_search.best_estimator_

        # Save search results
        cv_results = pd.DataFrame(random_search.cv_results_)
        cv_results_path = os.path.join(
            reports_dir, f"cv_results_{model_name}_{timestamp}.csv")
        cv_results.to_csv(cv_results_path, index=False)
    else:
        logger.info(f"Training {model_name} with base parameters...")
        model = multi_model
        model.fit(X_train, y_train)

    # Evaluation
    predictions = model.predict(X_test)
    f1 = f1_score(y_test, predictions, average='micro')
    precision = precision_score(y_test, predictions, average='micro')
    recall = recall_score(y_test, predictions, average='micro')

    logger.info(f"\nModel Performance Metrics:")
    logger.info(f"F1 micro: {f1:.4f}")
    logger.info(f"Precision micro: {precision:.4f}")
    logger.info(f"Recall micro: {recall:.4f}")

    # Save detailed report
    report = classification_report(y_test, predictions)
    logger.info("\nClassification Report:")
    logger.info(report)

    report_path = os.path.join(
        reports_dir, f"classification_report_{model_name}_{timestamp}.txt")
    with open(report_path, "w") as f:
        f.write(report)

    # Save the model with timestamp
    model_dir = 'models/supervised'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{model_run_name}.pkl")
    joblib.dump(model, model_path)

    return model, {
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
        'model_name': model_name,
        'timestamp': timestamp,
        'search_params': search_params,
        'best_params': random_search.best_params_ if search_params else None,
        'best_score': random_search.best_score_ if search_params else None
    }
