def Classifier_Optimizer(model_dict, X_train, y_train, scoring, n_iter=100, cv=5, randome_state=42, n_jobs=-1):
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
    randome_state : int, optional, default=42
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
        'logreg': {
            'logisticregression__C': ...
        }
    }

    # Loop through classifiers and perform RandomizedSearchCV
    for name, model in trained_model_dict.items():
        print(f"\nTraining {name}...")
        
        search = RandomizedSearchCV(
            model, 
            param_distributions=param_dist[name], 
            scoring=scoring, 
            n_iter=n_iter, 
            cv=cv, 
            random_state=random_state,
            n_jobs=n_jobs
        )
        
        search.fit(X_train, y_train)
        
        # Test the best model on the test set
        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        optimized_model_dict[name] = {
            'best_model': best_model,
            'best_params': search.best_params_
        }

        scoring_dict[name] = {
            'train_score': search.best_score_,
            'test_score': test_accuracy
        }
        
    return optimized_model_dict, scoring_dict
