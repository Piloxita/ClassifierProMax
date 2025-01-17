import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from classifierpromax.Feature_Selector import Feature_Selector

def test_Feature_Selector():
    # Mock dataset
    X_train = pd.DataFrame(np.random.rand(100, 5), columns=['A', 'B', 'C', 'D', 'E'])
    y_train = np.random.choice([0, 1], size=100)

    # Mock trained models
    trained_models = {
        "RandomForest": make_pipeline(RandomForestClassifier())
    }

    # 1. Valid RFE case
    result_rfe = Feature_Selector(X_train, y_train, trained_models, method='RFE', n_features_to_select=3)
    assert "RandomForest" in result_rfe
    assert hasattr(result_rfe["RandomForest"], 'steps')  # Check if a pipeline is returned

    # 2. Invalid method
    with pytest.raises(ValueError, match="Invalid feature selection method: InvalidMethod"):
        Feature_Selector(X_train, y_train, trained_models, method='InvalidMethod')

    # 3. Empty trained_models
    result_empty = Feature_Selector(X_train, y_train, {}, method='RFE', n_features_to_select=3)
    assert result_empty == {}

    # 4. Var Threshold method
    result_var = Feature_Selector(X_train, y_train, trained_models, method='Var Threshold')
    assert "RandomForest" in result_var

    # 5. Not implemented methods
    with pytest.raises(NotImplementedError, match="Backward SFS not implemented yet"):
        Feature_Selector(X_train, y_train, trained_models, method='Backward SFS')
    with pytest.raises(NotImplementedError, match="Forward SFS not implemented yet"):
        Feature_Selector(X_train, y_train, trained_models, method='Forward SFS')

