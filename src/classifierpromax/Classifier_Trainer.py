# Core Libraries
import os  # For file path operations

# Data Manipulation
import pandas as pd  # For handling DataFrame operations

# Machine Learning
from sklearn.dummy import DummyClassifier  # For dummy classification model
from sklearn.linear_model import LogisticRegression  # For logistic regression model
from sklearn.svm import SVC  # For support vector classifier
from sklearn.pipeline import make_pipeline  # For creating pipelines
from sklearn.ensemble import RandomForestClassifier  # For random forest classifier
from sklearn.model_selection import cross_validate  # For cross-validation

# Metrics and Scoring
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score  # For metrics

def Classifier_Trainer(preprocessor, X_train, y_train, pos_label, seed, cv=5, metrics=None):
    """
    Trains multiple classifiers on the provided dataset using cross-validation and returns a dictionary
    of trained models and a dictionary of scoring metrics.

    Parameters:
    -----------
    preprocessor : sklearn.pipeline.Pipeline or transformer
        A preprocessing pipeline to apply transformations to the training data.

    X_train : array-like of shape (n_samples, n_features)
        The training input data.

    y_train : array-like of shape (n_samples,)
        The target labels for training.

    pos_label : int or str
        The positive class label used for metrics calculation such as precision and recall.

    seed : int
        Random seed for reproducibility.

    cv : int, default=5
        Number of cross-validation folds.

    metrics : dict, optional
        A dictionary containing scoring metrics to use during evaluation. If None, defaults to accuracy,
        precision, recall, and f1 score.

    Returns:
    --------
    trained_model_dict : dict
        A dictionary where keys are model names and values are the corresponding trained models.

    scoring_dict : dict
        A dictionary where keys are model names and values are the corresponding scoring metrics calculated
        using the specified metrics during cross-validation.
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
            "precision": make_scorer(precision_score, pos_label=pos_label),
            "recall": make_scorer(recall_score, pos_label=pos_label),
            "f1": make_scorer(f1_score, pos_label=pos_label),
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
