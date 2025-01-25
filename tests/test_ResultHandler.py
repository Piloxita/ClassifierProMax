import pytest
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from classifierpromax.ResultHandler import ResultHandler

def test_result_handler_with_hyperparameters():
    models = {
        "logreg": LogisticRegression(random_state=42),
        "svc": SVC(random_state=42),
        "random_forest": RandomForestClassifier(random_state=42)
    }

    scoring_dict = {
        'logreg': {'accuracy': 0.85, 'precision': 0.82, 'recall': 0.84, 'f1': 0.83},
        'svc': {'accuracy': 0.87, 'precision': 0.86, 'recall': 0.88, 'f1': 0.87},
        'random_forest': {'accuracy': 0.89, 'precision': 0.87, 'recall': 0.88, 'f1': 0.87}
    }

    # Get the result from the ResultHandler
    result_df = ResultHandler(scoring_dict, models)

    # Check if expected index names are present in the result DataFrame
    expected_index = ['accuracy', 'precision', 'recall', 'f1', 'logreg__C']
    for idx in expected_index:
        assert idx in result_df.index, f"Index '{idx}' not found in the result DataFrame."

    # Check if the expected columns are present in the result DataFrame
    expected_columns = ['logreg', 'svc', 'random_forest']
    for col in expected_columns:
        assert col in result_df.columns, f"Column '{col}' not found in the result DataFrame."

def test_empty_scoring_dict_with_hyperparameters():
    scoring_dict = {}

    expected_df = pd.DataFrame(columns=[], index=['accuracy', 'precision', 'recall', 'f1'])

    result_df = ResultHandler(scoring_dict, models={})

    # Check if the expected index names are present in the result DataFrame (even if empty)
    expected_index = ['accuracy', 'precision', 'recall', 'f1']
    for idx in expected_index:
        assert idx in result_df.index, f"Index '{idx}' not found in the result DataFrame."

    # Check that there are no columns since scoring_dict is empty
    assert result_df.columns.empty, "Expected no columns in the result DataFrame."

