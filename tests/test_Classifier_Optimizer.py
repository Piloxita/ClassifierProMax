import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils.validation import check_is_fitted
from classifierpromax.Classifier_Optimizer import Classifier_Optimizer

@pytest.fixture
def setup_data():
    """Generate synthetic dataset and model dictionary for testing."""
    X, y = make_classification(
        n_samples=100, n_features=20, n_informative=15, n_redundant=5, random_state=42
    )
    model_dict = {
        'logreg': Pipeline([
            ('scaler', StandardScaler()),
            ('logisticregression', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        'svc': Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(probability=True, random_state=42))
        ])
    }
    return X, y, model_dict

def test_classifier_optimizer_valid_input(setup_data):
    """Test the function with valid input."""
    X, y, model_dict = setup_data
    scoring = 'accuracy'
    optimized_model_dict, scoring_dict = Classifier_Optimizer(
        model_dict, X, y, scoring=scoring, n_iter=10, cv=3, random_state=42, n_jobs=1
    )

    # Check if the function returns dictionaries
    assert isinstance(optimized_model_dict, dict)
    assert isinstance(scoring_dict, dict)

    # Check if all models are optimized
    assert all(name in optimized_model_dict for name in model_dict.keys())
    assert all(name in scoring_dict for name in model_dict.keys())

    # Check if models are fitted
    for name, model_info in optimized_model_dict.items():
        best_model = model_info['best_model']
        check_is_fitted(best_model)

    # Check scoring values are reasonable
    for scores in scoring_dict.values():
        assert 0 <= scores['accuracy_score'] <= 1
        assert 0 <= scores['f1_score'] <= 1
        assert 0 <= scores['precision_score'] <= 1
        assert 0 <= scores['recall_score'] <= 1

def test_classifier_optimizer_invalid_model_name(setup_data):
    """Test the function with an invalid model name in param_dist."""
    X, y, model_dict = setup_data
    model_dict['invalid_model'] = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC())
    ])
    
    with pytest.raises(KeyError):
        Classifier_Optimizer(model_dict, X, y, scoring='accuracy', n_iter=10, cv=3)

def test_classifier_optimizer_invalid_data(setup_data):
    """Test the function raises an error if both X_train and y_train are invalid."""
    _, y, model_dict = setup_data    
    with pytest.raises(ValueError, match="X_train must be a pandas DataFrame or a numpy array."):
        Classifier_Optimizer(model_dict, [], y, scoring='accuracy')

def test_classifier_optimizer_invalid_scoring(setup_data):
    """Test the function with an invalid scoring metric."""
    X, y, model_dict = setup_data
    with pytest.raises(ValueError):
        Classifier_Optimizer(model_dict, X, y, scoring='invalid_metric', n_iter=10, cv=3)

def test_classifier_optimizer_empty_model_dict(setup_data):
    """Test the function with an empty model dictionary."""
    X, y, _ = setup_data
    with pytest.raises(ValueError):
        Classifier_Optimizer({}, X, y, scoring='accuracy', n_iter=10, cv=3)

def test_classifier_optimizer_invalid_cv_value(setup_data):
    """Test the function with an invalid cv value."""
    X, y, model_dict = setup_data
    with pytest.raises(ValueError):
        Classifier_Optimizer(model_dict, X, y, scoring='accuracy', n_iter=10, cv=-1)