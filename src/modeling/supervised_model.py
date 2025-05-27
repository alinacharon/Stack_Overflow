import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV
import scipy.sparse

def get_base_model(model_name, model_params):
    if model_name == 'logreg':
        return LogisticRegression(**model_params)
    elif model_name == 'random_forest':
        return RandomForestClassifier(**model_params)
    elif model_name == 'xgboost':
        return XGBClassifier(use_label_encoder=False, **model_params)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def train_classifier(X_train, y_train, X_test, y_test, model_name='logreg', config=None):
    mlflow.set_tag("task", "multi-label classification")
    mlflow.log_param("model_name", model_name)
    
    model_config = config['supervised']['models'].get(model_name, {})
    base_params = model_config.get('base_params', {})
    search_params = model_config.get('search_params', None)

    base_model = get_base_model(model_name, base_params)
    multi_model = MultiOutputClassifier(base_model)

    def safe_cast(X):
        print(f"safe_cast input type: {type(X)}")
        if scipy.sparse.issparse(X):
            return X.astype(np.float32)
        elif isinstance(X, pd.DataFrame):
            return X.values.astype(np.float32)
        elif isinstance(X, np.ndarray):
            return X.astype(np.float32)
        else:
            raise TypeError(f"Unsupported input type: {type(X)}")

    def safe_cast_y(y):
        if isinstance(y, pd.DataFrame):
            return y.values.astype(np.int32)
        elif isinstance(y, np.ndarray):
            return y.astype(np.int32)
        else:
            return np.array(y, dtype=np.int32)

    X_train = safe_cast(X_train)
    X_test = safe_cast(X_test)
    y_train = safe_cast_y(y_train)
    y_test = safe_cast_y(y_test)

    if search_params:
        mlflow.set_tag("tuning", "RandomizedSearchCV")
        random_search = RandomizedSearchCV(
            estimator=multi_model,
            param_distributions=search_params,
            n_iter=10,
            cv=3,
            scoring='f1_micro',
            verbose=1,
            n_jobs=-1,
            random_state=42
        )
        random_search.fit(X_train, y_train)
        model = random_search.best_estimator_
        mlflow.log_params(random_search.best_params_)
    else:
        model = multi_model
        model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    f1 = f1_score(y_test, predictions, average='micro')
    precision = precision_score(y_test, predictions, average='micro')
    recall = recall_score(y_test, predictions, average='micro')

    print(f"F1 micro: {f1}")
    print(f"Precision micro: {precision}")
    print(f"Recall micro: {recall}")

    mlflow.log_metric("f1_micro", f1)
    mlflow.log_metric("precision_micro", precision)
    mlflow.log_metric("recall_micro", recall)

    report = classification_report(y_test, predictions)
    print(report)
    with open("classification_report.txt", "w") as f:
        f.write(report)
    mlflow.log_artifact("classification_report.txt")

    model_path = f"models/supervised/{model_name}.pkl"
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path)

    return model