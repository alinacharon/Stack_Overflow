import sys
import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
import joblib
from api.app import app, preprocess_text
from api.model_loader import load_model_and_vectorizer

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

# Set model paths
os.environ['MODEL_PATH'] = os.path.join(project_root, 'models/final/model.pkl')
os.environ['EMBEDDING_MODEL_PATH'] = os.path.join(
    project_root, 'models/final/embedding_model.pkl')
os.environ['MODEL_TYPE'] = os.getenv('MODEL_TYPE', 'supervised')


@pytest.fixture(scope="session")
def client():
    return TestClient(app)


@pytest.fixture(scope="session")
def model_and_embedding():
    """Load model and embedding model before tests"""
    model, embedding_model = load_model_and_vectorizer()
    if model is None or embedding_model is None:
        pytest.skip("Model files not found")
    return model, embedding_model


def test_health_check(client):
    """Test API health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_model_files_exist():
    """Test if model and embedding model files exist"""
    model_path = os.environ['MODEL_PATH']
    embedding_model_path = os.environ['EMBEDDING_MODEL_PATH']

    assert os.path.exists(model_path), f"Model file not found: {model_path}"
    assert os.path.exists(
        embedding_model_path), f"Embedding model file not found: {embedding_model_path}"


def test_text_preprocessing():
    """Test text preprocessing function"""
    test_cases = [
        ("Hello, World!", "hello world", "supervised"),
        ("Python 3.9", "python", "supervised"),
        ("   Extra   Spaces   ", "extra spaces", "supervised"),
        ("Special@#$%Chars", "specialchars", "supervised"),
        ("Mixed CASE", "mixed case", "supervised"),
        ("Hello, World!", "Hello, World!", "bert"),
        ("Python 3.9", "Python 3.9", "bert"),
        ("   Extra   Spaces   ", "Extra Spaces", "use"),
        ("Special@#$%Chars", "Special@#$%Chars", "use"),
        ("Mixed CASE", "Mixed CASE", "use")
    ]

    for input_text, expected, model_type in test_cases:
        assert preprocess_text(input_text, model_type) == expected


@pytest.mark.parametrize("question_text", [
    "How to implement a binary search tree in Python?",
    "What is machine learning?",
    "How to use pandas for data analysis?"
])
def test_model_prediction(client, question_text):
    """Test model predictions with different questions"""
    test_question = {"text": question_text}

    response = client.post("/predict", json=test_question)
    assert response.status_code == 200

    result = response.json()
    assert "tags" in result
    assert "probabilities" in result
    assert isinstance(result["tags"], list)
    assert isinstance(result["probabilities"], list)
    assert len(result["tags"]) == len(result["probabilities"])
    assert all(isinstance(tag, str) for tag in result["tags"])
    assert all(isinstance(prob, float) for prob in result["probabilities"])


def test_model_loading(model_and_embedding):
    """Test model and embedding model loading"""
    model, embedding_model = model_and_embedding
    assert model is not None, "Model not loaded"
    assert embedding_model is not None, "Embedding model not loaded"


def test_embedding_dimensions(model_and_embedding):
    """Test embedding model feature dimensions"""
    _, embedding_model = model_and_embedding
    model_type = os.getenv('MODEL_TYPE', 'supervised')

    # Test feature dimensions
    test_text = "How to implement a binary search tree in Python?"
    processed_text = preprocess_text(test_text, model_type)

    if model_type in ['bert', 'use']:
        X = embedding_model.get_embeddings([processed_text])
    else:
        X = embedding_model.transform([processed_text])

    assert X.shape[1] > 0, "Empty feature vector"


@pytest.mark.parametrize("invalid_input", [
    {"text": ""},  # Empty text
    {"text": "   "},  # Whitespace only
    {},  # Missing text field
    {"invalid_field": "some text"}  # Wrong field name
])
def test_invalid_inputs(client, invalid_input):
    """Test API behavior with invalid inputs"""
    response = client.post("/predict", json=invalid_input)
    # Bad Request or Validation Error
    assert response.status_code in [400, 422]


def test_predict(client):
    """Test prediction endpoint"""
    test_question = {
        "text": "How to fix ModuleNotFoundError: No module named 'pandas' error in Python?"
    }
    response = client.post("/predict", json=test_question)
    assert response.status_code == 200
    assert "tags" in response.json()
    assert "probabilities" in response.json()
    assert isinstance(response.json()["tags"], list)
    assert isinstance(response.json()["probabilities"], list)
    assert len(response.json()["tags"]) == len(
        response.json()["probabilities"])
