import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from classifierpromax.Feature_Selector import Feature_Selector

# Test setup: synthetic dataset
@pytest.fixture
def sample_data():
    X, y = make_classification(n_samples=100, n_features=10, n_informative=5, random_state=42)
    return X, y

# Test setup: models and pipelines
@pytest.fixture
def trained_models(sample_data):
    X, y = sample_data
    rf = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42))
    lr = make_pipeline(StandardScaler(), LogisticRegression(random_state=42))
    rf.fit(X, y)
    lr.fit(X, y)
    return {"RandomForest": rf, "LogisticRegression": lr}

# Test setup: preprocessor
@pytest.fixture
def preprocessor():
    return StandardScaler()

# Test RFE with valid input
def test_rfe_selection(sample_data, trained_models, preprocessor):
    X, y = sample_data
    selected_models = Feature_Selector(
        preprocessor, trained_models, X, y, method='RFE', n_features_to_select=5
    )
    assert "RandomForest" in selected_models
    assert "LogisticRegression" in selected_models
    assert len(selected_models["RandomForest"].steps) == 3  # Ensure pipeline has 3 steps

# Test Pearson feature selection
def test_pearson_selection(sample_data, trained_models, preprocessor):
    X, y = sample_data
    selected_models = Feature_Selector(
        preprocessor, trained_models, X, y, method='Pearson', n_features_to_select=5
    )
    assert "RandomForest" in selected_models
    assert "LogisticRegression" in selected_models
    assert len(selected_models["RandomForest"].steps) == 3  # Ensure pipeline has 3 steps

# Test invalid method
def test_invalid_method(sample_data, trained_models, preprocessor):
    X, y = sample_data
    with pytest.raises(ValueError, match="Invalid feature selection method: InvalidMethod"):
        Feature_Selector(
            preprocessor, trained_models, X, y, method='InvalidMethod'
        )

# Test missing n_features_to_select for RFE
def test_missing_n_features_to_select(sample_data, trained_models, preprocessor):
    X, y = sample_data
    with pytest.raises(ValueError, match="`n_features_to_select` must be provided for RFE."):
        Feature_Selector(
            preprocessor, trained_models, X, y, method='RFE'
        )
