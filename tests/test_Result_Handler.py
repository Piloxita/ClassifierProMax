import pytest
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from classifierpromax.Result_Handler import Result_Handler  # replace with the actual module name

@pytest.fixture
def scoring_dict_trainer():
    # Sample input data for scoring_dict_trainer
    data = {
        'model1': pd.DataFrame({'metric1': [0.9, 0.8], 'metric2': [0.85, 0.75]}),
        'model2': pd.DataFrame({'metric1': [0.95, 0.92], 'metric2': [0.88, 0.80]})
    }
    return data

@pytest.fixture
def scoring_dict_optimizer():
    # Sample input data for scoring_dict_optimizer
    data = {
        'model1': pd.DataFrame({'metric1': [0.91, 0.82], 'metric2': [0.86, 0.76]}),
        'model2': pd.DataFrame({'metric1': [0.96, 0.93], 'metric2': [0.89, 0.81]})
    }
    return data

def test_result_handler(scoring_dict_trainer, scoring_dict_optimizer):
    # Call the Result_Handler function
    result_df = Result_Handler(scoring_dict_trainer, scoring_dict_optimizer)
    
    # Verify the combined DataFrame structure
    assert isinstance(result_df, pd.DataFrame), "The result is not a DataFrame."
    
    # Verify the number of rows in the DataFrame
    assert result_df.shape[0] == 2, f"Expected 2 rows, but got {result_df.shape[0]}."
    
    # Verify the number of columns in the DataFrame
    expected_columns = pd.MultiIndex.from_tuples([
        ('model1', 'metric1'), ('model1', 'metric2'),
        ('model2', 'metric1'), ('model2', 'metric2')
    ])
    assert result_df.columns.equals(expected_columns), "The column structure is incorrect."
    
    # Check if index from optimizer is removed
    assert "index" not in result_df.columns.get_level_values(1), "The 'index' column should not be present."

    # Validate the values in the combined DataFrame
    expected_values = [
        [0.9, 0.85, 0.91, 0.86],
        [0.8, 0.75, 0.82, 0.76]
    ]
    pd.testing.assert_frame_equal(result_df.values, expected_values, check_dtype=False)