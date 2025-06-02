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

    if config['data'].get('use_sample', False):
        sample_config = config['data'].get('sample', {})
        if not isinstance(sample_config, dict):
            raise ValueError("Sample configuration must be a dictionary")

        required_sample_keys = ['size', 'top_n_tags', 'output_dir', 'seed']
        for key in required_sample_keys:
            if key not in sample_config:
                raise ValueError(
                    f"Missing required sample configuration key: {key}")

        if not isinstance(sample_config['size'], int) or sample_config['size'] <= 0:
            raise ValueError("Sample size must be a positive integer")

        if not isinstance(sample_config['top_n_tags'], int) or sample_config['top_n_tags'] <= 0:
            raise ValueError("top_n_tags must be a positive integer")

        if not isinstance(sample_config['seed'], int):
            raise ValueError("Seed must be an integer")

    if config['task_type'] == 'supervised':
        if 'supervised' not in config or 'model' not in config['supervised']:
            raise ValueError(
                "Missing supervised.model config for supervised task")

    print("🟢 Config verified!")
