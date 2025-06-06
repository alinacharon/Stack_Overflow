from sklearn.preprocessing import MultiLabelBinarizer
import joblib
import os


def fit_and_binarize(tags, save_path='models/supervised/mlb.pkl'):
    """Fit MultiLabelBinarizer and transform tags to binary format"""
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(tags)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(mlb, save_path)
    print(f'MultiLabelBinarizer saved to {save_path}')

    columns = mlb.classes_
    return Y, mlb, columns