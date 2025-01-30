import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from classifierpromax.ClassifierOptimizer import ClassifierOptimizer

# ---------------------
# Fixtures
# ---------------------

@pytest.fixture
def sample_data():
    """Generate a synthetic dataset with 100 samples and 5 features."""
    np.random.seed(42)
    X_train = np.random.rand(100, 5)
    y_train = np.random.randint(0, 2, size=100)
    return X_train, y_train

@pytest.fixture
def test_models():
    """Define a dictionary of test models with preprocessing pipelines."""
    return {
        'logreg': Pipeline([
            ('scaler', StandardScaler()),
            ('logisticregression', LogisticRegression())
        ]),
        'svc': Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC())
        ]),
        'random_forest': Pipeline([
            ('randomforestclassifier', RandomForestClassifier())
        ])
    }

# ---------------------
# Test Cases
# ---------------------

def test_classifier_optimizer_invalid_model_dict(sample_data):
    """Ensure ValueError is raised for a non-dictionary model input."""
    X_train, y_train = sample_data
    invalid_model_dict = ["not", "a", "dict"]

    with pytest.raises(ValueError, match="model_dict must be a non-empty dictionary of sklearn Pipeline objects."):
        ClassifierOptimizer(invalid_model_dict, X_train, y_train)

def test_classifier_optimizer_empty_model_dict(sample_data):
    """Ensure ValueError is raised when an empty model dictionary is provided."""
    X_train, y_train = sample_data

    with pytest.raises(ValueError, match="model_dict must be a non-empty dictionary of sklearn Pipeline objects."):
        ClassifierOptimizer({}, X_train, y_train)

def test_classifier_optimizer_invalid_model_names(sample_data):
    """Ensure ValueError is raised for non-string model names."""
    X_train, y_train = sample_data
    invalid_model_dict = {123: Pipeline([("scaler", StandardScaler()), ("logreg", LogisticRegression())])}

    with pytest.raises(ValueError, match="Invalid model name '123'. Model names must be non-empty strings."):
        ClassifierOptimizer(invalid_model_dict, X_train, y_train)

def test_classifier_optimizer_invalid_model_type(sample_data):
    """Ensure ValueError is raised when a model dictionary contains non-Pipeline objects."""
    X_train, y_train = sample_data
    invalid_model_dict = {"invalid_model": "not_a_pipeline"}

    with pytest.raises(ValueError, match="The model 'invalid_model' is not a valid scikit-learn Pipeline."):
        ClassifierOptimizer(invalid_model_dict, X_train, y_train)

def test_classifier_optimizer_invalid_X_train(sample_data, test_models):
    """Ensure ValueError is raised when X_train is not a DataFrame or ndarray."""
    X_train, y_train = "invalid_input", sample_data[1]

    with pytest.raises(ValueError, match="X_train must be a pandas DataFrame or a numpy array."):
        ClassifierOptimizer(test_models, X_train, y_train)

def test_classifier_optimizer_invalid_y_train(sample_data, test_models):
    """Ensure ValueError is raised when y_train is not a Series or ndarray."""
    X_train, y_train = sample_data[0], "invalid_labels"

    with pytest.raises(ValueError, match="y_train must be a pandas Series or a numpy array."):
        ClassifierOptimizer(test_models, X_train, y_train)

def test_classifier_optimizer_empty_X_train(test_models):
    """Ensure ValueError is raised when X_train is empty."""
    X_train = np.array([]).reshape(0, 5)
    y_train = np.random.randint(0, 2, size=10)

    with pytest.raises(ValueError, match="X_train cannot be empty."):
        ClassifierOptimizer(test_models, X_train, y_train)

def test_classifier_optimizer_empty_y_train(test_models):
    """Ensure ValueError is raised when y_train is empty."""
    X_train = np.random.rand(10, 5)
    y_train = np.array([])

    with pytest.raises(ValueError, match="y_train cannot be empty."):
        ClassifierOptimizer(test_models, X_train, y_train)

def test_classifier_optimizer_mismatched_X_y(test_models):
    """Ensure ValueError is raised when X_train and y_train have different sample sizes."""
    X_train = np.random.rand(50, 5)
    y_train = np.random.randint(0, 2, size=100)

    with pytest.raises(ValueError, match="The number of samples in X_train and y_train must match."):
        ClassifierOptimizer(test_models, X_train, y_train)

def test_classifier_optimizer_invalid_scoring(sample_data, test_models):
    """Ensure ValueError is raised when an invalid scoring metric is provided."""
    X_train, y_train = sample_data

    with pytest.raises(ValueError, match="Invalid scoring metric 'invalid_score'. Choose from .*"):
        ClassifierOptimizer(test_models, X_train, y_train, scoring="invalid_score")

@pytest.mark.parametrize("n_iter", [0, -5])
def test_classifier_optimizer_invalid_n_iter(sample_data, test_models, n_iter):
    """Ensure ValueError is raised when n_iter is non-positive."""
    X_train, y_train = sample_data

    with pytest.raises(ValueError, match="n_iter must be a positive integer."):
        ClassifierOptimizer(test_models, X_train, y_train, n_iter=n_iter)

def test_classifier_optimizer_invalid_cv(sample_data, test_models):
    """Ensure ValueError is raised when cv is too low."""
    X_train, y_train = sample_data

    with pytest.raises(ValueError, match="cv must be an integer greater than 1."):
        ClassifierOptimizer(test_models, X_train, y_train, cv=1)

@pytest.mark.parametrize("random_state", ["invalid", 3.14, None])
def test_classifier_optimizer_invalid_random_state(sample_data, test_models, random_state):
    """Ensure ValueError is raised when random_state is not an integer."""
    X_train, y_train = sample_data

    with pytest.raises(ValueError, match="random_state must be an integer."):
        ClassifierOptimizer(test_models, X_train, y_train, random_state=random_state)

def test_classifier_optimizer_invalid_n_jobs(sample_data, test_models):
    """Ensure ValueError is raised when n_jobs is zero."""
    X_train, y_train = sample_data

    with pytest.raises(ValueError, match="n_jobs must be a nonzero integer .*"):
        ClassifierOptimizer(test_models, X_train, y_train, n_jobs=0)

@pytest.mark.parametrize("scoring_metric", ["accuracy", "precision", "recall", "f1"])
def test_classifier_optimizer_different_scoring(sample_data, test_models, scoring_metric):
    """Test ClassifierOptimizer with different scoring metrics."""
    X_train, y_train = sample_data

    optimized_models, scoring_results = ClassifierOptimizer(
        test_models, X_train, y_train, scoring=scoring_metric, n_iter=5, cv=3, random_state=42, n_jobs=1
    )

    assert isinstance(optimized_models, dict)
    assert isinstance(scoring_results, dict)
    assert set(optimized_models.keys()) == set(test_models.keys())

    for model_name, scores in scoring_results.items():
        assert isinstance(scores, pd.DataFrame)
        assert any(scoring_metric in str(index) for index in scores.index), f"Scoring metric '{scoring_metric}' not found in results"
