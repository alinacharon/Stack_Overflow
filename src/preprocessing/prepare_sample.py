from collections import Counter
import logging
import pandas as pd
import os

logger = logging.getLogger(__name__)


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


def load_data(x_path, y_path):
    """Loading data from CSV files"""
    logger.info(f"Loading data from:\n - {x_path}\n - {y_path}")
    X = pd.read_csv(x_path)
    Y = pd.read_csv(y_path, converters={'Tags': eval})
    return X, Y


def create_sample(config):
    """Create sample from full dataset using configuration parameters"""
    # Load full dataset
    X, Y = load_data(config['data']['train_path'],
                     config['data']['target_path'])

    # Get top N tags
    top_tags = get_top_tags(Y, config['data']['sample']['top_n_tags'])
    logger.info(f"Selected top {len(top_tags)} tags")

    # Filter data by top tags
    X_filtered, Y_filtered = filter_by_top_tags(X, Y, top_tags)
    logger.info(f"Data after filtering by top tags: {X_filtered.shape}")

    # Create sample
    X_sample, Y_sample = sample_data(
        X_filtered,
        Y_filtered,
        config['data']['sample']['size'],
        config['data']['sample']['seed']
    )
    logger.info(f"Created sample of size: {X_sample.shape}")

    # Save sample
    save_sample(X_sample, Y_sample, config['data']['sample']['output_dir'])

    return X_sample, Y_sample
