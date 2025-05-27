import yaml

def validate_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    required_keys = ['task_type', 'mlflow', 'data', 'vectorizer']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required top-level key: {key}")

    if config['task_type'] not in ['lda', 'supervised']:
        raise ValueError(f"Unsupported task_type: {config['task_type']}")


    if config['task_type'] == 'supervised':
        if 'supervised' not in config or 'model' not in config['supervised']:
            raise ValueError("Missing supervised.model config for supervised task")

    print("🟢 Config verified!")