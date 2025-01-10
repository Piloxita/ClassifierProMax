def Classifier_Trainer(preprocessor, X_train, y_train, pos_lable, seed, cv = 5, metrics = None):
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

    pos_lable : int or str
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

    Note:
    -----
    The models included in the training are:
    - Dummy Classifier
    - Logistic Regression
    - Support Vector Classifier (SVC)
    - Logistic Regression with balanced class weights
    - Support Vector Classifier (SVC) with balanced class weights

    Example Usage:
    --------------
    trained_models, scores = Classifier_Trainer(preprocessor, X_train, y_train, pos_label=1, seed=42)
    """

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

