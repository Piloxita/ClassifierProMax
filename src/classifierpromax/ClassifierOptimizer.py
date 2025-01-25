import pandas as pd
import numpy as np
from scipy.stats import uniform, loguniform, randint
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, cross_validate
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, make_scorer

def ClassifierOptimizer(model_dict, X_train, y_train, scoring='f1', n_iter=100, cv=5, random_state=42, n_jobs=-1):
    """
    Perform hyperparameter optimization for multiple classification models using RandomizedSearchCV.

    Parameters:
    ----------
    model_dict : dict
        A dictionary containing the names and corresponding classifier pipeline models.
    X_train : array-like or DataFrame
        Training feature set.
    y_train : array-like or Series
        Training target labels.
    scoring : str or callable
        Metric used to evaluate model performance (e.g., 'accuracy', 'f1', etc.).
    n_iter : int, optional, default=100
        Number of parameter settings sampled for RandomizedSearchCV.
    cv : int, optional, default=5
        Number of cross-validation folds.
    random_state : int, optional, default=42
        Random seed to ensure reproducibility.
    n_jobs : int, optional, default=-1
        Number of CPU cores used for parallel processing. -1 uses all available cores.

    Returns:
    -------
    optimized_model_dict : dict
        A dictionary containing the optimized models and their best parameters.
    scoring_dict : dict
        A dictionary containing training and test scores for each classifier.
    """

    param_dist = {
        'logreg' : {
            'logisticregression__C': loguniform(1e-2, 1e3),
            'logisticregression__class_weight': [None, 'balanced']
        },
        'svc' : {
            'svc__C': loguniform(1e-2, 1e3),
            'svc__class_weight': [None, 'balanced']
        },
        'random_forest' : {
            'randomforestclassifier__n_estimators': randint(10,50),
            'randomforestclassifier__max_depth': randint(5,15)
        }
    }

    # Default metrics if not provided
    scoring_metrics = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0, average='weighted'),
        "recall": make_scorer(recall_score, average='weighted'),
        "f1": make_scorer(f1_score, average='weighted'),
    }

    # Validate scoring
    if scoring not in scoring_metrics:
        raise ValueError(f"Invalid scoring metric '{scoring}'. Choose from {list(scoring_metrics.keys())}.")
    
    # Validate model_dict
    if not isinstance(model_dict, dict):
        raise ValueError("model_dict must be a dictionary.")
    if not model_dict:
        raise ValueError("model_dict is empty. Please provide at least one model.")

    for name, model in model_dict.items():
        if not isinstance(model, Pipeline):
            raise ValueError(f"The model '{name}' is not a valid scikit-learn Pipeline.")

    # Validate X_train, y_train
    if not isinstance(X_train, (pd.DataFrame, np.ndarray)):
        raise ValueError("X_train must be a pandas DataFrame or a numpy array.")
    if not isinstance(y_train, (pd.Series, np.ndarray)):
        raise ValueError("y_train must be a pandas Series or a numpy array.")
    if X_train.size == 0 or y_train.size == 0:
        raise ValueError("X_train and y_train cannot be empty.")
    if X_train.shape[0] != len(y_train):
        raise ValueError("The number of samples in X_train and y_train must match.")
    
    # Check if the model names match the param_dist keys
    if not all(name in param_dist for name in model_dict):
        raise ValueError("Each model name in model_dict must have corresponding hyperparameters in param_dist.")

    # Drop dummy model
    model_dict.pop('dummy', None)

    optimized_model_dict = {}
    scoring_dict = {}

    # Loop through classifiers and perform RandomizedSearchCV
    for name, model in model_dict.items():
        print(f"\nTraining {name}...")
        
        search = RandomizedSearchCV(
            estimator=model, 
            param_distributions=param_dist[name], 
            scoring=scoring_metrics[scoring], 
            n_iter=n_iter, 
            cv=cv, 
            random_state=random_state,
            n_jobs=n_jobs,
            return_train_score=True
        )
        
        search.fit(X_train, y_train)
        search.best_params_
        
        best_model = search.best_estimator_
        optimized_model_dict[name] = best_model

        cv_results = cross_validate(
            best_model, 
            X_train, 
            y_train, 
            cv=cv, 
            scoring=scoring_metrics, 
            return_train_score=True,
            error_score='raise'
        )
        scoring_dict[name] = pd.DataFrame(cv_results).agg(['mean', 'std']).T
    return optimized_model_dict, scoring_dict
