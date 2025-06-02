import mlflow
import yaml
import os


def setup_mlflow(config_path='config/config.yaml'):

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    mlflow_config = config.get('mlflow', {})

    tracking_uri = mlflow_config.get('tracking_uri', 'sqlite:///mlflow.db')
    mlflow.set_tracking_uri(tracking_uri)

    registry_uri = mlflow_config.get('registry_uri', 'sqlite:///mlflow.db')
    mlflow.set_registry_uri(registry_uri)

    experiment_name = mlflow_config.get('experiment_name', 'default')
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            name=experiment_name,
            artifact_location=mlflow_config.get(
                'artifacts_location', './mlruns')
        )
    else:
        experiment_id = experiment.experiment_id

    return experiment_id