import pytest
import pandas as pd
from classifierpromax.ResultHandler import ResultHandler

# ---------------------
# Test Cases
# ---------------------

def test_result_handler_valid_input():
    """Ensure ResultHandler correctly processes valid input with and without std."""
    scoring_dict_trainer = {
        "model1": pd.DataFrame({"mean": [0.85], "std": [0.03]}),
        "model2": pd.DataFrame({"mean": [0.80], "std": [0.04]})
    }
    scoring_dict_optimizer = {
        "model1": pd.DataFrame({"mean": [0.88], "std": [0.02]}),
        "model2": pd.DataFrame({"mean": [0.83], "std": [0.03]})
    }

    # Case with std=False
    result = ResultHandler(scoring_dict_trainer, scoring_dict_optimizer, std=False)
    assert isinstance(result, pd.DataFrame)
    assert "model1_baseline" in result.columns
    assert "model1_optimized" in result.columns
    assert result.loc[:, "model1_baseline"].iloc[0] == 0.85

    # Case with std=True
    result = ResultHandler(scoring_dict_trainer, scoring_dict_optimizer, std=True)
    assert isinstance(result, pd.DataFrame)
    assert ("mean" in result.columns.get_level_values(1))
    assert ("std" in result.columns.get_level_values(1))

def test_result_handler_without_optimizer():
    """Ensure ResultHandler works when no optimizer results are provided."""
    scoring_dict_trainer = {
        "model1": pd.DataFrame({"mean": [0.85], "std": [0.03]}),
        "model2": pd.DataFrame({"mean": [0.80], "std": [0.04]})
    }

    result = ResultHandler(scoring_dict_trainer)

    assert isinstance(result, pd.DataFrame)
    assert "model1" in result.columns
    assert "model2" in result.columns

def test_result_handler_invalid_trainer_input():
    """Ensure ValueError is raised when scoring_dict_trainer is not a dictionary."""
    with pytest.raises(ValueError, match="scoring_dict_trainer must be a dictionary"):
        ResultHandler(scoring_dict_trainer="invalid_input")

def test_result_handler_invalid_optimizer_input():
    """Ensure ValueError is raised when scoring_dict_optimizer is not a dictionary."""
    scoring_dict_trainer = {
        "model1": pd.DataFrame({"mean": [0.85], "std": [0.03]})
    }
    with pytest.raises(ValueError, match="scoring_dict_optimizer must be a dictionary"):
        ResultHandler(scoring_dict_trainer, scoring_dict_optimizer="invalid_input")

def test_result_handler_invalid_dataframe():
    """Ensure ValueError is raised when a dictionary value is not a DataFrame."""
    scoring_dict_trainer = {
        "model1": "invalid_df"
    }
    with pytest.raises(ValueError, match="Value for key 'model1' in scoring_dict_trainer must be a pandas DataFrame"):
        ResultHandler(scoring_dict_trainer)

def test_result_handler_invalid_std_input():
    """Ensure ValueError is raised when std is not a boolean."""
    scoring_dict_trainer = {
        "model1": pd.DataFrame({"mean": [0.85], "std": [0.03]})
    }
    with pytest.raises(ValueError, match="std must be a boolean value"):
        ResultHandler(scoring_dict_trainer, std="invalid")
