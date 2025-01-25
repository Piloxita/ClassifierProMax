import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from classifierpromax.ClassifierOptimizer import ClassifierOptimizer

@pytest.fixture
def sample_data():
    # Generate synthetic data for testing
    X = pd.DataFrame(np.random.rand(100, 5), columns=[f"feature_{i}" for i in range(5)])
    y = pd.Series(np.random.randint(0, 2, size=100))
    return X, y

@pytest.fixture
def sample_models():
    # Create a dictionary of pipelines for testing
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
            ('scaler', StandardScaler()),
            ('randomforestclassifier', RandomForestClassifier())
        ])
    }

def test_classifier_optimizer_valid_input(sample_data, sample_models):
    X, y = sample_data
    model_dict = sample_models
    scoring = 'accuracy'
    
    optimized_models, scoring_dict = ClassifierOptimizer(
        model_dict=model_dict,
        X_train=X,
        y_train=y,
        scoring=scoring,
        n_iter=5,
        cv=2
    )
    
    assert isinstance(optimized_models, dict)
    assert isinstance(scoring_dict, dict)
    for name in model_dict.keys():
        assert name in optimized_models
        assert name in scoring_dict

def test_classifier_optimizer_invalid_model_dict(sample_data):
    X, y = sample_data
    invalid_model_dict = "not_a_dict"
    with pytest.raises(ValueError, match="model_dict must be a dictionary."):
        ClassifierOptimizer(model_dict=invalid_model_dict, X_train=X, y_train=y, scoring='accuracy')

def test_classifier_optimizer_empty_model_dict(sample_data):
    X, y = sample_data
    empty_model_dict = {}
    with pytest.raises(ValueError, match="model_dict is empty. Please provide at least one model."):
        ClassifierOptimizer(model_dict=empty_model_dict, X_train=X, y_train=y, scoring='accuracy')

def test_classifier_optimizer_invalid_X_train(sample_models):
    model_dict = sample_models
    X_train = "not_a_dataframe"
    y_train = pd.Series(np.random.randint(0, 2, size=100))
    with pytest.raises(ValueError, match="X_train must be a pandas DataFrame or a numpy array."):
        ClassifierOptimizer(model_dict=model_dict, X_train=X_train, y_train=y_train, scoring='accuracy')

def test_classifier_optimizer_invalid_y_train(sample_models):
    model_dict = sample_models
    X_train = pd.DataFrame(np.random.rand(100, 5))
    y_train = "not_a_series"
    with pytest.raises(ValueError, match="y_train must be a pandas Series or a numpy array."):
        ClassifierOptimizer(model_dict=model_dict, X_train=X_train, y_train=y_train, scoring='accuracy')

def test_classifier_optimizer_mismatched_X_y_lengths(sample_models):
    model_dict = sample_models
    X_train = pd.DataFrame(np.random.rand(100, 5))
    y_train = pd.Series(np.random.randint(0, 2, size=50))  # Mismatched length
    with pytest.raises(ValueError, match="The number of samples in X_train and y_train must match."):
        ClassifierOptimizer(model_dict=model_dict, X_train=X_train, y_train=y_train, scoring='accuracy')