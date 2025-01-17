import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from classifierpromax.Feature_Selector import Feature_Selector

# Helper function to create a mock dataset
def create_mock_data():
    X_train = pd.DataFrame(np.random.rand(100, 5), columns=['A', 'B', 'C', 'D', 'E'])
    y_train = np.random.choice([0, 1], size=100)
    return X_train, y_train

# Test for a valid RFE case
def test_feature_selector_rfe():
    X_train, y_train = create_mock_data()
    trained_models = {
        "RandomForest": make_pipeline(RandomForestClassifier())
    }
    result_rfe = Feature_Selector(X_train, y_train, trained_models, method='RFE', n_features_to_select=3)
    assert "RandomForest" in result_rfe
    assert hasattr(result_rfe["RandomForest"], 'steps')  # Check if a pipeline is returned

# Test for invalid method
def test_feature_selector_invalid_method():
    X_train, y_train = create_mock_data()
    trained_models = {
        "RandomForest": make_pipeline(RandomForestClassifier())
    }
    with pytest.raises(ValueError, match="Invalid feature selection method: InvalidMethod"):
        Feature_Selector(X_train, y_train, trained_models, method='InvalidMethod')

# Test for empty trained_models
def test_feature_selector_empty_trained_models():
    X_train, y_train = create_mock_data()
    result_empty = Feature_Selector(X_train, y_train, {}, method='RFE', n_features_to_select=3)
    assert result_empty == {}

# Test for variance threshold method
def test_feature_selector_var_threshold():
    X_train, y_train = create_mock_data()
    trained_models = {
        "RandomForest": make_pipeline(RandomForestClassifier())
    }
    result_var = Feature_Selector(X_train, y_train, trained_models, method='Var Threshold')
    assert "RandomForest" in result_var

# Test for not implemented methods
def test_feature_selector_not_implemented_methods():
    X_train, y_train = create_mock_data()
    trained_models = {
        "RandomForest": make_pipeline(RandomForestClassifier())
    }
    with pytest.raises(NotImplementedError, match="Backward SFS not implemented yet"):
        Feature_Selector(X_train, y_train, trained_models, method='Backward SFS')
    with pytest.raises(NotImplementedError, match="Forward SFS not implemented yet"):
        Feature_Selector(X_train, y_train, trained_models, method='Forward SFS')
