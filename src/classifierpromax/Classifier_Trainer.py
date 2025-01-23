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

def Classifier_Trainer(preprocessor, X_train, y_train, pos_label, seed, cv=5, metrics=None):
    """
    Train multiple machine learning classifiers using cross-validation and return trained models
    and evaluation metrics.

    This function applies a preprocessing pipeline to the training data, trains several classifiers
    using cross-validation, and evaluates their performance based on specified or default metrics.
    It supports handling common validation checks and provides a structured summary of model scores.

    Parameters:
    -----------
    preprocessor : sklearn.pipeline.Pipeline or transformer
        A preprocessing pipeline or transformer object with `fit` and `transform` methods. 
        Used to preprocess the training data.

    X_train : array-like of shape (n_samples, n_features)
        Training input data. Each row represents a sample, and each column represents a feature.

    y_train : array-like of shape (n_samples,)
        Target labels corresponding to the training data.

    pos_label : int or str
        The positive class label used for calculating metrics like precision, recall, and F1 score.

    seed : int
        Random seed for ensuring reproducibility in model training and evaluation.

    cv : int, default=5
        Number of folds for cross-validation. Determines how the data is split for training and validation.

    metrics : dict, optional
        A dictionary where keys are metric names (strings) and values are either strings representing
        predefined metrics (e.g., "accuracy") or callable scoring functions (e.g., `make_scorer`).
        If None, the default metrics include accuracy, precision, recall, and F1 score.

    Returns:
    --------
    trained_model_dict : dict
        A dictionary containing the trained models. Keys are model names (e.g., "logreg", "svc"), and
        values are the trained model objects.

    scoring_dict : dict
        A dictionary containing evaluation metrics for each model. Keys are model names, and values
        are pandas DataFrames summarizing the mean and standard deviation of each metric across
        cross-validation folds.
    """
    # Validate preprocessor
    if not all(hasattr(preprocessor, method) for method in ["fit", "transform"]):
        raise TypeError("The preprocessor must have 'fit' and 'transform' methods.")

    # Validate shapes of X_train and y_train
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            f"Mismatch between the number of samples in X_train ({X_train.shape[0]}) "
            f"and y_train ({y_train.shape[0]}). They must be the same."
        )

    # Validate positive label
    if pos_label not in y_train:
        raise ValueError(
            f"The specified positive label '{pos_label}' is not found in y_train. "
            f"Ensure that the label is present in the dataset."
        )

    # Default metrics if not provided
    if metrics is None:
        metrics = {
            "accuracy": "accuracy",
            "precision": make_scorer(precision_score, pos_label=pos_label, zero_division=0, average='weighted'),
            "recall": make_scorer(recall_score, pos_label=pos_label, average='weighted'),
            "f1": make_scorer(f1_score, pos_label=pos_label, average='weighted'),
        }

    # Define classifiers
    models = {
        "dummy": make_pipeline(preprocessor, DummyClassifier(strategy="most_frequent")),
        "logreg": make_pipeline(preprocessor, LogisticRegression(random_state=seed, max_iter=1000)),
        "svc": make_pipeline(preprocessor, SVC(kernel='linear', random_state=seed)),
        "random_forest": make_pipeline(preprocessor, RandomForestClassifier(random_state=seed)),
    }

    trained_model_dict = {}
    scoring_dict = {}

    # Train and evaluate models
    for model_name, pipeline in models.items():
        cv_results = cross_validate(
            pipeline, 
            X_train, 
            y_train, 
            cv=cv, 
            scoring=metrics, 
            return_train_score=True,
            error_score='raise'
        )
        trained_model_dict[model_name] = pipeline.fit(X_train, y_train)
        scoring_dict[model_name] = pd.DataFrame(cv_results).agg(['mean', 'std']).T

    return trained_model_dict, scoring_dict
