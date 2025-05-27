import os
import joblib
from sklearn.preprocessing import MultiLabelBinarizer


def fit_and_binarize(tags, save_path='models/supervised/mlb.pkl'):
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(tags)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(mlb, save_path)
    print(f'MultiLabelBinarizer saved to {save_path}')

    columns = mlb.classes_
    return Y, mlb, columns


def load_binarizer(path='models/supervised/mlb.pkl'):
    """
    Load a saved MultiLabelBinarizer from disk.
    """
    return joblib.load(path)