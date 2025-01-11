def Feature_Selector(X_train, y_train, trained_models, method='RFE', scoring='accuracy', n_features_to_select=None):
  """
  Selects features for multiple classification models using various methods.
  Args:
      X_train (array-like or DataFrame): Training feature set.
      y_train (array-like or Series): Training target labels.
      trained_models (dict): A dictionary containing the names and corresponding trained classification models.
      method (str, optional): Feature selection method. Defaults to 'RFE'. Can be 'RFE', 'Backward SFS', 'Forward SFS', 'Var Threshold', or 'Pearson'.
      scoring (str, optional): Scoring metric used for model selection during feature selection. Defaults to 'accuracy'.
      n_features_to_select (int, optional): The number of features to select (for methods like RFE). If None, all features are used for model selection. Defaults to None.
  Returns:
      dict: A dictionary containing the feature selected models. Keys are model names and values are pipelines with feature selection applied.
  Raises:
      ValueError: If an invalid method is provided.
  """
  feature_selected_models = {}
  for model_name, model in trained_models.items():
    if method == 'RFE':
      # Using Recursive feature elimination method
      selector = RFE(model, n_features_to_select=n_features_to_select)
      new_model = make_pipeline(selector, model.steps[-1][1])  
      feature_selected_models[model_name] = new_model
    elif method == 'Backward SFS':
      # Implement Backward Sequential Feature Selection using cross-validation
      raise NotImplementedError("Backward SFS not implemented yet")
    elif method == 'Forward SFS':
      # Implement Forward Sequential Feature Selection using cross-validation 
      raise NotImplementedError("Forward SFS not implemented yet")
    elif method == 'Var Threshold':
      # Use feature selection based on variance
      from sklearn.feature_selection import VarianceThreshold
      selector = VarianceThreshold()
      new_model = make_pipeline(selector, model.steps[-1][1])  
      feature_selected_models[model_name] = new_model
    elif method == 'Pearson':
      # feature selection with Pearson correlation coefficient
      from sklearn.feature_selection import SelectKBest, f_classif
      selector = SelectKBest(f_classif, k=n_features_to_select) 
      new_model = make_pipeline(selector, model.steps[-1][1])  
      feature_selected_models[model_name] = new_model
    else:
      raise ValueError(f"Invalid feature selection method: {method}")
  return feature_selected_models