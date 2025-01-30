import pandas as pd
import numpy as np
from scipy.stats import uniform, loguniform, randint
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, cross_validate
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, make_scorer

# Helper function: Define hyperparameter distributions
def get_param_distributions():
    return {
        'logreg': {
            'logisticregression__C': loguniform(1e-2, 1e3),
            'logisticregression__class_weight': [None, 'balanced']
        },
        'svc': {
            'svc__C': loguniform(1e-2, 1e3),
            'svc__class_weight': [None, 'balanced']
        },
        'random_forest': {
            'randomforestclassifier__n_estimators': randint(10, 30),
            'randomforestclassifier__max_depth': randint(5, 10)
        }
    }

# Helper function: Define scoring metrics
def get_scoring_metrics():
    return {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0, average='weighted'),
        "recall": make_scorer(recall_score, average='weighted'),
        "f1": make_scorer(f1_score, average='weighted'),
    }

# Helper function: Validate inputs
def validate_inputs(model_dict, param_dist, X_train, y_train, scoring, n_iter, cv, random_state, n_jobs):

    # Validate model_dict
    if not isinstance(model_dict, dict) or not model_dict:
        raise ValueError("model_dict must be a non-empty dictionary of sklearn Pipeline objects.")

    for name, model in model_dict.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"Invalid model name '{name}'. Model names must be non-empty strings.")
        if not isinstance(model, Pipeline):
            raise ValueError(f"The model '{name}' is not a valid scikit-learn Pipeline.")

    # Validate X_train
    if not isinstance(X_train, (pd.DataFrame, np.ndarray)):
        raise ValueError("X_train must be a pandas DataFrame or a numpy array.")
    if X_train.size == 0:
        raise ValueError("X_train cannot be empty.")

    # Validate y_train
    if not isinstance(y_train, (pd.Series, np.ndarray)):
        raise ValueError("y_train must be a pandas Series or a numpy array.")
    if y_train.size == 0:
        raise ValueError("y_train cannot be empty.")

    # Validate consistency of X_train and y_train
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError("The number of samples in X_train and y_train must match.")

    # Validate scoring metric
    valid_metrics = get_scoring_metrics().keys()
    if scoring not in valid_metrics:
        raise ValueError(f"Invalid scoring metric '{scoring}'. Choose from {list(valid_metrics)}.")

    # Validate numeric parameters
    if not isinstance(n_iter, int) or n_iter <= 0:
        raise ValueError("n_iter must be a positive integer.")
    if not isinstance(cv, int) or cv <= 1:
        raise ValueError("cv must be an integer greater than 1.")
    if not isinstance(random_state, int):
        raise ValueError("random_state must be an integer.")
    if not isinstance(n_jobs, int) or n_jobs == 0:
        raise ValueError("n_jobs must be a nonzero integer (use -1 for all processors).")

# Helper function: Optimize a single model
def optimize_model(name, model, param_dist, X_train, y_train, scoring, n_iter, cv, random_state, n_jobs):
    print(f"\nTraining {name}...")
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        scoring=get_scoring_metrics()[scoring],
        n_iter=n_iter,
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
        return_train_score=True
    )
    search.fit(X_train, y_train)
    print(f"Best parameters for {name}: {search.best_params_}")
    return search.best_estimator_

# Helper function: Evaluate model performance
def evaluate_model(name, model, X_train, y_train, cv):
    cv_results = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=get_scoring_metrics(),
        return_train_score=True,
        error_score='raise'
    )
    return pd.DataFrame(cv_results).agg(['mean', 'std']).T

# Main function: ClassifierOptimizer
def ClassifierOptimizer(model_dict, X_train, y_train, scoring='f1', n_iter=100, cv=5, random_state=42, n_jobs=-1):
    """
    Optimizes a dictionary of scikit-learn Pipeline classifiers using RandomizedSearchCV 
    and evaluates their performance.

    Parameters:
    -----------
    model_dict : dict
        A dictionary where keys are model names (str) and values are scikit-learn Pipeline objects.
        Each pipeline must contain a classifier whose hyperparameters are defined in `param_dist`.
    
    X_train : pandas.DataFrame or numpy.ndarray
        The feature matrix for training the classifiers. Must have the same number of samples as `y_train`.
    
    y_train : pandas.Series or numpy.ndarray
        The target labels for training the classifiers. Must have the same number of samples as `X_train`.

    scoring : str, optional, default='f1'
        The scoring metric to use for hyperparameter optimization and model evaluation. 
        Must be one of the following:
        - "accuracy"
        - "precision"
        - "recall"
        - "f1"

    n_iter : int, optional, default=100
        The number of parameter settings sampled for RandomizedSearchCV.

    cv : int, optional, default=5
        The number of cross-validation folds for both RandomizedSearchCV and cross_validate.

    random_state : int, optional, default=42
        Random seed for reproducibility of RandomizedSearchCV.

    n_jobs : int, optional, default=-1
        The number of jobs to run in parallel for RandomizedSearchCV (-1 uses all available processors).

    Returns:
    --------
    optimized_model_dict : dict
        A dictionary containing the best estimators for each classifier after hyperparameter optimization.

    scoring_dict : dict
        A dictionary containing cross-validation results for each optimized model, with 
        metrics aggregated by mean and standard deviation.

    Raises:
    -------
    ValueError
        If the input parameters are invalid (e.g., empty model dictionary, mismatched data shapes, 
        unsupported scoring metric).

    Examples:
    ---------
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.svm import SVC
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.preprocessing import StandardScaler
    >>> model_dict = {
    ...     'logreg': Pipeline([
    ...         ('scaler', StandardScaler()),
    ...         ('logisticregression', LogisticRegression())
    ...     ]),
    ...     'svc': Pipeline([
    ...         ('scaler', StandardScaler()),
    ...         ('svc', SVC())
    ...     ]),
    ...     'random_forest': Pipeline([
    ...         ('randomforestclassifier', RandomForestClassifier())
    ...     ])
    ... }
    >>> optimized_models, scoring_results = ClassifierOptimizer(model_dict, X_train, y_train)
    """
    
    param_dist = get_param_distributions()
    validate_inputs(model_dict, param_dist, X_train, y_train, scoring, n_iter, cv, random_state, n_jobs)
    
    optimized_model_dict = {}
    scoring_dict = {}
    
    for name, model in model_dict.items():
        best_model = optimize_model(
            name, model, param_dist[name], X_train, y_train, scoring, n_iter, cv, random_state, n_jobs
        )
        optimized_model_dict[name] = best_model
        scoring_dict[name] = evaluate_model(name, best_model, X_train, y_train, cv)
    
    return optimized_model_dict, scoring_dict