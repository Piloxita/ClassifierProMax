import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from classifierpromax.FeatureSelector import FeatureSelector
# Fixture: Synthetic Dataset
@pytest.fixture
def sample_data():
    """
    Provides a synthetic dataset for testing with 100 samples as well as 10 features.
    """
    X, y = make_classification(n_samples=100, n_features=10, n_informative=5, random_state=42)
    return X, y
# Fixture: Trained Models
@pytest.fixture
def trained_models(sample_data):
    """
    Returns pipelines with RandomForest and Logistic Regression models trained on sample data.
    """
    X, y = sample_data
    rf = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42))
    lr = make_pipeline(StandardScaler(), LogisticRegression(random_state=42))
    rf.fit(X, y)
    lr.fit(X, y)
    return {"RandomForest": rf, "LogisticRegression": lr}
# Fixture: Preprocessor
@pytest.fixture
def preprocessor():
    """
    Returns a standard scaler as the preprocessing step.
    """
    return StandardScaler()
### Test 1: RFE with Different Number of Features
@pytest.mark.parametrize("n_features", [2, 5, 7])
def test_rfe_selection(sample_data, trained_models, preprocessor, n_features):
    """
    Test RFE with different numbers of features to select.
    """
    X, y = sample_data
    selected_models = FeatureSelector(preprocessor, trained_models, X, y, method='RFE', n_features_to_select=n_features)
    assert "RandomForest" in selected_models
    assert "LogisticRegression" in selected_models
    assert len(selected_models["RandomForest"].steps) == 3  # Ensure the pipeline has preprocessor, selector, and classifier.
### Test 2: Pearson with Different Feature Thresholds
@pytest.mark.parametrize("n_features", [3, 5, 8])
def test_pearson_selection(sample_data, trained_models, preprocessor, n_features):
    """
    Test Pearson feature selection (SelectKBest) with varying thresholds for feature count.
    """
    X, y = sample_data
    selected_models = FeatureSelector(preprocessor, trained_models, X, y, method='Pearson', n_features_to_select=n_features)
    assert "RandomForest" in selected_models
    assert "LogisticRegression" in selected_models
    assert len(selected_models["RandomForest"].steps) == 3
### Test 3: Invalid Method Exception
def test_invalid_method(sample_data, trained_models, preprocessor):
    """
    Test if an invalid method raises a ValueError.
    """
    X, y = sample_data
    with pytest.raises(ValueError, match="Invalid feature selection method: InvalidMethod"):
        FeatureSelector(preprocessor, trained_models, X, y, method='InvalidMethod')
### Test 4: Missing n_features_to_select for RFE
def test_missing_n_features_rfe(sample_data, trained_models, preprocessor):
    """
    Test if missing n_features_to_select for RFE raises a ValueError.
    """
    X, y = sample_data
    with pytest.raises(ValueError, match="`n_features_to_select` must be provided for RFE."):
        FeatureSelector(preprocessor, trained_models, X, y, method='RFE')
### Test 5: Edge Case Pearson Selection with All Features
def test_pearson_all_features(sample_data, trained_models, preprocessor):
    """
    Test Pearson selection with all features selected (edge case).
    """
    X, y = sample_data
    selected_models = FeatureSelector(preprocessor, trained_models, X, y, method='Pearson', n_features_to_select=X.shape[1])
    assert "RandomForest" in selected_models
    assert "LogisticRegression" in selected_models



