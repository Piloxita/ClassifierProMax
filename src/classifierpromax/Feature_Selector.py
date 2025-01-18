from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import RFE, VarianceThreshold, SelectKBest, f_classif

def Feature_Selector(X_train, y_train, trained_models, preprocessor, method='RFE', scoring='accuracy', n_features_to_select=None):
    """
    Selects features for multiple classification models using various methods.
    Args:
        X_train (array-like or DataFrame): Training feature set.
        y_train (array-like or Series): Training target labels.
        trained_models (dict): A dictionary containing the names and corresponding trained best classification models.
        preprocessor (Pipeline or Transformer): Preprocessing pipeline to include in the final pipeline.
        method (str, optional): Feature selection method. Defaults to 'RFE'. Can be 'RFE', 'Backward SFS', 'Forward SFS', 'Var Threshold', or 'Pearson'.
        scoring (str, optional): Scoring metric used for model selection during feature selection. Defaults to 'accuracy'.
        n_features_to_select (int, optional): The number of features to select (for methods like RFE). Defaults to None.
    Returns:
        dict: A dictionary containing the feature-selected models. Keys are model names and values are pipelines with feature selection applied.
    Raises:
        ValueError: If an invalid method is provided or required parameters are missing.
    """
    feature_selected_models = {}

    for model_name, model in trained_models.items():
        # Extract the base estimator from the pipeline
        base_model = model.steps[-1][1]

        if method == 'RFE':
            if n_features_to_select is None:
                raise ValueError("`n_features_to_select` must be provided for RFE.")
            # Apply RFE
            selector = RFE(base_model, n_features_to_select=n_features_to_select)
            selector.fit(X_train, y_train)
            # Create a new pipeline with the preprocessor, selector, and base model
            new_model = make_pipeline(preprocessor, selector, base_model)
            feature_selected_models[model_name] = new_model

        elif method == 'Var Threshold':
            # Apply VarianceThreshold
            selector = VarianceThreshold(threshold=0.0)  # Adjust threshold if needed
            selector.fit(X_train)
            new_model = make_pipeline(preprocessor, selector, base_model)
            feature_selected_models[model_name] = new_model

        elif method == 'Pearson':
            if n_features_to_select is None:
                raise ValueError("`n_features_to_select` must be provided for Pearson method.")
            # Use SelectKBest
            selector = SelectKBest(f_classif, k=n_features_to_select)
            selector.fit(X_train, y_train)
            new_model = make_pipeline(preprocessor, selector, base_model)
            feature_selected_models[model_name] = new_model

        elif method in ['Backward SFS', 'Forward SFS']:
            raise NotImplementedError(f"{method} is not implemented yet.")

        else:
            raise ValueError(f"Invalid feature selection method: {method}")

    return feature_selected_models
