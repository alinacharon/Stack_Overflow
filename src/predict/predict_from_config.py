import pandas as pd
import joblib
import yaml
from sklearn.model_selection import train_test_split
import mlflow

def load_config(path='config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def predict_from_config(config_path='config.yaml'):
    config = load_config(config_path)

    # --- Пути ---
    data_path = config['data_path']
    tags_path = config['classifier']['multilabel_targets_path']
    model_path = config['classifier']['model_path']
    vectorizer_path = 'models/supervised/vectorizer.pkl'
    mlb_path = 'models/supervised/mlb.pkl'

    # --- Загрузка текста и таргета ---
    X_df = pd.read_csv(data_path)
    texts = X_df.iloc[:, 0].astype(str).tolist()

    y_df = pd.read_csv(tags_path)

    # --- Загрузка векторайзера и трансформация ---
    vectorizer = joblib.load(vectorizer_path)
    X = vectorizer.transform(texts)

    # --- Повторный сплит для извлечения теста ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_df, test_size=0.2, random_state=42
    )

    # --- Загрузка модели и MultiLabelBinarizer ---
    model = joblib.load(model_path)
    mlb = joblib.load(mlb_path)

    # --- Предсказание ---
    y_pred_bin = model.predict(X_test)
    y_pred_tags = mlb.inverse_transform(y_pred_bin)
    y_true_tags = mlb.inverse_transform(y_test.values)

    # --- Тексты тестовой выборки ---
    test_texts = [texts[i] for i in y_test.index]

    # --- Результат в DataFrame ---
    df_pred = pd.DataFrame({
        'text': test_texts,
        'true_tags': y_true_tags,
        'predicted_tags': y_pred_tags
    })

    # --- Лог в MLflow (опционально) ---
    experiment_name = config['mlflow']['experiment_name']
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name='predict_manual'):
        mlflow.log_param('task_type', config['task_type'])
        mlflow.log_param('model_name', config['classifier']['model_name'])
        mlflow.log_param('vectorizer', config['vectorizer']['method'])
        mlflow.log_metric('num_predictions', len(df_pred))

    return df_pred