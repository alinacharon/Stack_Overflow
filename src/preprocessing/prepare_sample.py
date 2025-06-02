# preprocessing/prepare_sample.py

import pandas as pd
from collections import Counter
import yaml
import os
import logging

logger = logging.getLogger(__name__)


def load_config(config_path='config/config.yaml'):
    """Loading configuration from yaml file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_data(x_path, y_path):
    """Loading data from CSV files"""
    X = pd.read_csv(x_path)
    Y = pd.read_csv(y_path, converters={'Tags': eval})
    return X, Y


def get_top_tags(Y, top_n):
    """Getting top N tags from data"""
    all_tags = [tag for tags in Y['Tags'] for tag in tags]
    tag_counts = Counter(all_tags)
    return [tag for tag, _ in tag_counts.most_common(top_n)]


def filter_by_top_tags(X, Y, top_tags):
    """Filtering data by top tags"""
    Y['Tags'] = Y['Tags'].apply(
        lambda tags: [tag for tag in tags if tag in top_tags])
    mask = Y['Tags'].map(len) > 0
    return X[mask].reset_index(drop=True), Y[mask].reset_index(drop=True)


def sample_data(X, Y, sample_size, seed):
    """Creating sample of data"""
    sampled_idx = X.sample(n=min(sample_size, len(X)), random_state=seed).index
    return X.loc[sampled_idx].reset_index(drop=True), Y.loc[sampled_idx].reset_index(drop=True)


def save_sample(X_sample, Y_sample, out_dir):
    """Saving sample to CSV files"""
    os.makedirs(out_dir, exist_ok=True)
    x_path = os.path.join(out_dir, 'X_sample.csv')
    y_path = os.path.join(out_dir, 'Y_sample.csv')
    X_sample.to_csv(x_path, index=False)
    Y_sample.to_csv(y_path, index=False)
    logger.info(
        f"Sample saved to:\n - {x_path} ({X_sample.shape})\n - {y_path} ({Y_sample.shape})")


def prepare_sample(config_path='config/config.yaml'):
    """Main function for preparing sample of data"""
    config = load_config(config_path)

    if not config['data'].get('use_sample', False):
        logger.info("Sample preparation skipped: use_sample is False")
        return None, None

    sample_config = config['data'].get('sample', {})
    x_path = config['data']['train_path']
    y_path = config['data']['target_path']

    logger.info("Loading data...")
    X, Y = load_data(x_path, y_path)

    logger.info(f"Getting top {sample_config['top_n_tags']} tags...")
    top_tags = get_top_tags(Y, sample_config['top_n_tags'])

    logger.info("Filtering by top tags...")
    X_filtered, Y_filtered = filter_by_top_tags(X, Y, top_tags)

    logger.info(f"Creating sample of size {sample_config['size']}...")
    X_sample, Y_sample = sample_data(
        X_filtered,
        Y_filtered,
        sample_config['size'],
        sample_config['seed']
    )

    logger.info("Saving sample...")
    save_sample(X_sample, Y_sample, sample_config['output_dir'])

    return X_sample, Y_sample


if __name__ == '__main__':
    # Logging setup
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run sample preparation
    prepare_sample()
