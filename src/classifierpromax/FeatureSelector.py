from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import RFE, SelectKBest, f_classif

def FeatureSelector(preprocessor, trained_models, X_train, y_train, method='RFE', n_features_to_select=None):
    """
    Selects features for multiple classification models using RFE or Pearson methods.

    Parameters: 
    -----------
    preprocessor : sklearn.pipeline.Pipeline or Transformer
        Preprocessing pipeline to include in the final pipeline.

    trained_models : dict
        A dictionary containing the names and corresponding trained best classification models. 
        Keys are model names, and values are trained pipelines.

    X_train : array-like or DataFrame
        Training feature set.

    y_train : array-like or Series
        Training target labels.

    method : str, optional
        Feature selection method. Defaults to 'RFE'. 
        Can be one of {'RFE', 'Pearson'}.

    n_features_to_select : int, optional
        The number of features to select. 
        Required for both 'RFE' and 'Pearson' methods. Defaults to None.

    Returns:
    --------
    feature_selected_models : dict
        A dictionary containing the feature-selected models. 
        Keys are model names, and values are pipelines with feature selection applied.

    Raises:
    -------
    ValueError
        If `n_features_to_select` is not provided or an invalid method is specified.

    Examples:
    ---------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.datasets import make_classification
    >>> X_train, y_train = make_classification(n_samples=100, n_features=10, random_state=42)
    >>> trained_models = {
    ...     'RandomForest': make_pipeline(StandardScaler(), RandomForestClassifier())
    ... }
    >>> preprocessor = StandardScaler()
    >>> feature_selected_models = FeatureSelector(
    ...     preprocessor, trained_models, X_train, y_train, method='RFE', n_features_to_select=5
    ... )
    >>> print(feature_selected_models.keys())
    dict_keys(['RandomForest'])
    """
    feature_selected_models = {}

    # Drop dummy model
    trained_models.pop('dummy', None)

    for model_name, model in trained_models.items():
        # Extract the base estimator from the pipeline
        base_model = model.steps[-1][1]

        if method == 'RFE':
            if n_features_to_select is None:
                raise ValueError("`n_features_to_select` must be provided for RFE.")
            # Apply RFE
            selector = RFE(base_model, n_features_to_select=n_features_to_select)
            # Create a new pipeline with the preprocessor, selector, and base model
            new_model = make_pipeline(preprocessor, selector, base_model)
            new_model.fit(X_train, y_train)
            feature_selected_models[model_name] = new_model

        elif method == 'Pearson':
            if n_features_to_select is None:
                raise ValueError("`n_features_to_select` must be provided for Pearson method.")
            # Use SelectKBest
            selector = SelectKBest(f_classif, k=n_features_to_select)
            new_model = make_pipeline(preprocessor, selector, base_model)
            new_model.fit(X_train, y_train)
            feature_selected_models[model_name] = new_model

        else:
            raise ValueError(f"Invalid feature selection method: {method}")

    return feature_selected_models

