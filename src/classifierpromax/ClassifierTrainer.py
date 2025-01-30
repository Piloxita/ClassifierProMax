# Core Libraries
import os
import pandas as pd

# Machine Learning
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

# Metrics and Scoring
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

def validate_preprocessor(preprocessor):
    """
    Validate if the preprocessor has the required `fit` and `transform` methods.

    Raises
    ------
    TypeError
        If the preprocessor does not have 'fit' and 'transform' methods.

    Example
    -------
    >>> from sklearn.preprocessing import StandardScaler
    >>> validate_preprocessor(StandardScaler())  # No error
    >>> validate_preprocessor("invalid")  # Raises TypeError
    """
    if not all(hasattr(preprocessor, method) for method in ["fit", "transform"]):
        raise TypeError("The preprocessor must have 'fit' and 'transform' methods.")

def validate_data(X_train, y_train):
    """
    Ensure that X_train and y_train have the same number of samples.

    Raises
    ------
    ValueError
        If the number of samples in X_train and y_train do not match.

    Example
    -------
    >>> import numpy as np
    >>> X_train = np.random.rand(100, 5)
    >>> y_train = np.random.randint(0, 2, size=100)
    >>> validate_data(X_train, y_train)  # No error
    >>> y_train = np.random.randint(0, 2, size=50)
    >>> validate_data(X_train, y_train)  # Raises ValueError
    """
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            f"Mismatch between X_train samples ({X_train.shape[0]}) and y_train samples ({y_train.shape[0]})."
        )

def get_scoring_metrics():
    """
    Return a dictionary of default scoring metrics for model evaluation.

    Returns
    -------
    dict
        A dictionary containing accuracy, precision, recall, and F1-score.

    Example
    -------
    >>> metrics = get_scoring_metrics()
    >>> print(metrics.keys())  # dict_keys(['accuracy', 'precision', 'recall', 'f1'])
    """
    return {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0, average='weighted'),
        "recall": make_scorer(recall_score, average='weighted'),
        "f1": make_scorer(f1_score, average='weighted'),
    }

def define_models(preprocessor, seed):
    """
    Define a set of machine learning models with preprocessing pipelines.

    Parameters
    ----------
    preprocessor : sklearn.pipeline.Pipeline or transformer
        Preprocessing pipeline applied before training the models.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        A dictionary containing model names as keys and pipelines as values.

    Example
    -------
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.pipeline import Pipeline
    >>> preprocessor = Pipeline([("scaler", StandardScaler())])
    >>> models = define_models(preprocessor, seed=42)
    >>> print(models.keys())  # dict_keys(['dummy', 'logreg', 'svc', 'random_forest'])
    """
    return {
        "dummy": make_pipeline(preprocessor, DummyClassifier(strategy="most_frequent")),
        "logreg": make_pipeline(preprocessor, LogisticRegression(random_state=seed, max_iter=1000)),
        "svc": make_pipeline(preprocessor, SVC(kernel='linear', random_state=seed)),
        "random_forest": make_pipeline(preprocessor, RandomForestClassifier(random_state=seed)),
    }

def train_and_evaluate(models, X_train, y_train, cv, metrics):
    """
    Train models using cross-validation and return trained models with their evaluation metrics.

    Parameters
    ----------
    models : dict
        Dictionary of model pipelines to train.
    X_train : array-like
        Training input data.
    y_train : array-like
        Training target labels.
    cv : int
        Number of cross-validation folds.
    metrics : dict
        Scoring metrics for evaluation.

    Returns
    -------
    trained_models : dict
        Dictionary containing trained models.
    scoring_results : dict
        Dictionary of DataFrames summarizing mean and std of cross-validation scores.

    Raises
    ------
    ValueError
        If training fails due to misaligned data.

    Example
    -------
    >>> from sklearn.datasets import make_classification
    >>> X_train, y_train = make_classification(n_samples=100, n_features=5, random_state=42)
    >>> preprocessor = Pipeline([("scaler", StandardScaler())])
    >>> models = define_models(preprocessor, seed=42)
    >>> trained_models, scoring_results = train_and_evaluate(models, X_train, y_train, cv=5, metrics=get_scoring_metrics())
    >>> print(scoring_results['logreg'])  # DataFrame with mean/std scores
    """
    trained_models = {}
    scoring_results = {}

    for model_name, pipeline in models.items():
        cv_results = cross_validate(
            pipeline, X_train, y_train, cv=cv, scoring=metrics, return_train_score=True, error_score='raise'
        )
        trained_models[model_name] = pipeline.fit(X_train, y_train)
        scoring_results[model_name] = pd.DataFrame(cv_results).agg(['mean', 'std']).T

    return trained_models, scoring_results

def ClassifierTrainer(preprocessor, X_train, y_train, seed, cv=5, metrics=None):
    """
    Train multiple machine learning classifiers using cross-validation.

    Parameters
    ----------
    preprocessor : sklearn.pipeline.Pipeline or transformer
        A preprocessing pipeline or transformer object with `fit` and `transform` methods.
    X_train : array-like
        Training input data.
    y_train : array-like
        Target labels corresponding to the training data.
    seed : int
        Random seed for reproducibility.
    cv : int, default=5
        Number of folds for cross-validation.
    metrics : dict, optional
        Dictionary of scoring metrics. If None, default metrics are used.

    Returns
    -------
    trained_models : dict
        Dictionary of trained model instances.
    scoring_results : dict
        Dictionary of evaluation results as pandas DataFrames.

    Raises
    ------
    TypeError
        If preprocessor is not a valid transformer.
    ValueError
        If X_train and y_train have inconsistent sample sizes.

    Example
    -------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> X_train, y_train = make_classification(n_samples=100, n_features=5, random_state=42)
    >>> preprocessor = Pipeline([("scaler", StandardScaler())])
    >>> seed = 42
    >>> trained_models, scoring_results = ClassifierTrainer(preprocessor, X_train, y_train, seed)
    >>> print(scoring_results['logreg'])  # DataFrame with mean/std scores
    """
    # Validate inputs
    validate_preprocessor(preprocessor)
    validate_data(X_train, y_train)

    # Get default or custom metrics
    metrics = metrics or get_scoring_metrics()

    # Define models
    models = define_models(preprocessor, seed)

    # Train models and collect evaluation results
    return train_and_evaluate(models, X_train, y_train, cv, metrics)
