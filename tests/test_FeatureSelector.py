import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from classifierpromax.FeatureSelector import FeatureSelector

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
@pytest.mark.parametrize("n_features", [3, 5, 7])
def test_rfe_different_features(sample_data, trained_models, preprocessor, n_features):
    X, y = sample_data
    selected_models = FeatureSelector(
        preprocessor, trained_models, X, y, method='RFE', n_features_to_select=n_features
    )
    assert "RandomForest" in selected_models
    assert "LogisticRegression" in selected_models

# Test Pearson with different thresholds
@pytest.mark.parametrize("threshold", [0.1, 0.3, 0.5])
def test_pearson_different_thresholds(sample_data, trained_models, preprocessor, threshold):
    X, y = sample_data
    selected_models = FeatureSelector(
        preprocessor, trained_models, X, y, method='Pearson', threshold=threshold
    )
    assert "RandomForest" in selected_models
    assert "LogisticRegression" in selected_models

# Test invalid method
def test_invalid_method(sample_data, trained_models, preprocessor):
    X, y = sample_data
    with pytest.raises(ValueError, match="Invalid feature selection method: InvalidMethod"):
        FeatureSelector(
            preprocessor, trained_models, X, y, method='InvalidMethod'
        )

# Test missing n_features_to_select for RFE
def test_missing_n_features_to_select(sample_data, trained_models, preprocessor):
    X, y = sample_data
    with pytest.raises(ValueError, match="`n_features_to_select` must be provided for RFE."):
        FeatureSelector(
            preprocessor, trained_models, X, y, method='RFE'
        )

# Test RFE when n_features_to_select exceeds available features
def test_rfe_exceeding_features(sample_data, trained_models, preprocessor):
    X, y = sample_data
    with pytest.raises(ValueError, match="n_features_to_select cannot exceed the number of features."):
        FeatureSelector(
            preprocessor, trained_models, X, y, method='RFE', n_features_to_select=15
        )

# Test Pearson when no features meet the threshold
def test_pearson_no_features_selected(sample_data, trained_models, preprocessor):
    X, y = sample_data
    selected_models = FeatureSelector(
        preprocessor, trained_models, X, y, method='Pearson', threshold=1.0
    )
    assert all(model.steps[-1][0] == 'classifier' for model in selected_models.values())

# Test empty model dictionary
def test_empty_model_dictionary(sample_data, preprocessor):
    X, y = sample_data
    with pytest.raises(ValueError, match="No models provided for feature selection."):
        FeatureSelector(preprocessor, {}, X, y, method='RFE', n_features_to_select=5)
