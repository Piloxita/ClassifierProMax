import pytest
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.exceptions import UndefinedMetricWarning
from classifierpromax.ClassifierTrainer import ClassifierTrainer

import warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# ---------------------
# Fixtures
# ---------------------

@pytest.fixture
def sample_data():
    """Generate a synthetic classification dataset for testing."""
    X, y = make_classification(n_samples=100, n_features=10, n_informative=5, n_classes=2, random_state=42)
    return X, y

@pytest.fixture
def preprocessor():
    """Create a basic preprocessing pipeline for numerical features."""
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    preprocessor = ColumnTransformer(
        transformers=[('num', numeric_transformer, slice(0, 10))]  # All 10 features are numeric
    )
    return preprocessor

# ---------------------
# Expanded Test Cases
# ---------------------

@pytest.mark.parametrize("invalid_preprocessor", [None, "not_a_pipeline", 123, [], {}])
def test_invalid_preprocessor(sample_data, invalid_preprocessor):
    """Ensure ClassifierTrainer raises TypeError for various invalid preprocessors."""
    X, y = sample_data
    seed = 42

    with pytest.raises(TypeError):
        ClassifierTrainer(invalid_preprocessor, X, y, seed)

@pytest.mark.parametrize("x_size, y_size", [(100, 90), (100, 110), (50, 100), (0, 100)])
def test_mismatched_shapes(preprocessor, x_size, y_size):
    """Ensure ClassifierTrainer raises ValueError when X and y sizes don't match."""
    X = np.random.rand(x_size, 10)
    y = np.random.randint(0, 2, size=y_size)
    seed = 42

    with pytest.raises(ValueError):
        ClassifierTrainer(preprocessor, X, y, seed)

@pytest.mark.parametrize("custom_metrics", [
    {"precision": make_scorer(precision_score, zero_division=0, average='weighted')},
    {"recall": make_scorer(recall_score, average='weighted')},
    {"f1": make_scorer(f1_score, average='weighted')},
    {},
])
def test_custom_metrics(sample_data, preprocessor, custom_metrics):
    """Test ClassifierTrainer with different metric configurations."""
    X, y = sample_data
    seed = 42

    _, scoring_dict = ClassifierTrainer(preprocessor, X, y, seed, metrics=custom_metrics)

    assert isinstance(scoring_dict, dict)
    for model_name, scores in scoring_dict.items():
        for metric in custom_metrics.keys():
            metric_name = f"test_{metric}" if metric else "test_accuracy"
            assert metric_name in scores.index

@pytest.mark.parametrize("data_type", ["numpy", "pandas"])
def test_different_data_types(sample_data, preprocessor, data_type):
    """Ensure ClassifierTrainer works with both NumPy arrays and pandas DataFrames."""
    X, y = sample_data
    seed = 42

    if data_type == "pandas":
        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        y = pd.Series(y)

    trained_model_dict, scoring_dict = ClassifierTrainer(preprocessor, X, y, seed)

    assert isinstance(trained_model_dict, dict)
    assert isinstance(scoring_dict, dict)

@pytest.mark.parametrize("X_empty, y_empty", [(np.array([]).reshape(0, 10), np.array([])), (np.array([]), np.array([]))])
def test_empty_input(preprocessor, X_empty, y_empty):
    """Ensure ClassifierTrainer raises an error for empty inputs."""
    seed = 42

    with pytest.raises(ValueError):
        ClassifierTrainer(preprocessor, X_empty, y_empty, seed)
