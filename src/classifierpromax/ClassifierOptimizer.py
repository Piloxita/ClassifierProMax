import pandas as pd
import numpy as np
from scipy.stats import uniform, loguniform, randint
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, cross_validate
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, make_scorer

def ClassifierOptimizer(model_dict, X_train, y_train, scoring='f1', n_iter=100, cv=5, random_state=42, n_jobs=-1):
    """
    Optimizes a dictionary of scikit-learn Pipeline classifiers using RandomizedSearchCV and evaluates their performance.

    Parameters
    ----------
    model_dict : dict
        A dictionary where keys are model names (str) and values are scikit-learn Pipeline objects.
        Each pipeline must contain a classifier whose hyperparameters are defined in `param_dist`.

    X_train : pandas.DataFrame or numpy.ndarray
        The feature matrix for training the classifiers. Must have the same number of samples as `y_train`.

    y_train : pandas.Series or numpy.ndarray
        The target labels for training the classifiers. Must have the same number of samples as `X_train`.

    scoring : dict, optional
        A dictionary specifying scoring metrics to evaluate the classifiers during cross-validation. 
        Default is None, which uses the following metrics:
            - "accuracy"
            - "precision" (weighted)
            - "recall" (weighted)
            - "f1" (weighted)

    n_iter : int, optional
        The number of parameter settings sampled for RandomizedSearchCV. Default is 100.

    cv : int, optional
        The number of cross-validation folds for both RandomizedSearchCV and cross_validate. Default is 5.

    random_state : int, optional
        Random seed for reproducibility of RandomizedSearchCV. Default is 42.

    n_jobs : int, optional
        The number of jobs to run in parallel for RandomizedSearchCV. Default is -1 (use all available processors).

    Returns
    -------
    optimized_model_dict : dict
        A dictionary containing the best estimators for each classifier after hyperparameter optimization.

    scoring_dict : dict
        A dictionary containing cross-validation results for each optimized model, with metrics aggregated by mean and standard deviation.

    Examples
    --------
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
    >>> optimized_models, scoring_results = Classifier_Optimizer(model_dict, X_train, y_train)
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
            'randomforestclassifier__max_depth': randint(5,10)
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
