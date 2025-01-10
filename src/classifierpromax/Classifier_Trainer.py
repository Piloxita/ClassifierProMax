def Classifier_Trainer(preprocessor, X_train, y_train, pos_lable, seed, cv = 5, metrics = None):

    models = {
        "dummy": make_pipeline(DummyClassifier()),
        "logreg": make_pipeline(preprocessor, LogisticRegression(random_state=seed, max_iter=1000)),
        "svc": make_pipeline(preprocessor, SVC(random_state=seed)),
        "logreg_bal": make_pipeline(preprocessor, LogisticRegression(random_state=seed, max_iter=1000, class_weight="balanced")),
        "svc_bal": make_pipeline(preprocessor, SVC(random_state=seed, class_weight="balanced"))
    }

     metrics = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, pos_label=pos_lable),
        "recall": make_scorer(recall_score, pos_label=pos_lable),
        "f1": make_scorer(f1_score, pos_label=pos_lable),
    }

    trained_model_dict = {}
    scoring_dict = {}

    return trained_model_dict, scoring_dict

