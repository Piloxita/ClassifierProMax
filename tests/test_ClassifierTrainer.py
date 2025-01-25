import pytest
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from classifierpromax.ClassifierTrainer import ClassifierTrainer

# Metrics and Scoring
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score  # For metrics


# Suppress UndefinedMetricWarning of dummy
import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Generate a synthetic dataset for testing
@pytest.fixture
def sample_data():
    X, y = make_classification(
        n_samples=100, n_features=10, n_informative=5, n_classes=2, random_state=42
    )
    return X, y

@pytest.fixture
def preprocessor():
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, slice(0, 10))  # All 10 features are numeric
        ]
    )

    return preprocessor

def test_valid_input(sample_data, preprocessor):
    X, y = sample_data
    seed = 42

    trained_model_dict, scoring_dict = ClassifierTrainer(
        preprocessor, X, y, seed
    )

    assert isinstance(trained_model_dict, dict)
    assert isinstance(scoring_dict, dict)
    assert set(trained_model_dict.keys()) == {"dummy", "logreg", "svc", "random_forest"}
    assert set(scoring_dict.keys()) == {"dummy", "logreg", "svc", "random_forest"}

def test_invalid_preprocessor(sample_data):
    X, y = sample_data
    seed = 42
    invalid_preprocessor = "not_a_pipeline"

    with pytest.raises(TypeError):
        ClassifierTrainer(invalid_preprocessor, X, y, seed)

def test_mismatched_shapes(preprocessor):
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, size=90)  # Mismatched length
    seed = 42

    with pytest.raises(ValueError):
        ClassifierTrainer(preprocessor, X, y, seed)

def test_custom_metrics(sample_data, preprocessor):
    X, y = sample_data
    seed = 42

    custom_metrics = {
        "precision": make_scorer(precision_score, zero_division=0, average='weighted'),
        "recall": make_scorer(recall_score, average='weighted'),
        "f1": make_scorer(f1_score, average='weighted'),
    }

    _, scoring_dict = ClassifierTrainer(preprocessor, X, y, seed, metrics=custom_metrics)

    for model_name, scores in scoring_dict.items():
        assert "test_precision" in scores.index
        assert "train_recall" in scores.index
        assert "test_f1" in scores.index
