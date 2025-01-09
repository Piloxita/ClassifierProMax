def Classifier_Optimizer(model_dict, X_train, y_train, scoring, n_iter=100, cv=5, randome_state=42, n_jobs=-1):

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
