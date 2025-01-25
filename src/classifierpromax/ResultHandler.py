import pandas as pd

def ResultHandler(scoring_dict_trainer, scoring_dict_optimizer=None, std=False):
    """
    Processes and combines scoring results from model training and optimization.

    Parameters:
    -----------
    scoring_dict_trainer : dict
        A dictionary where keys are model names and values are DataFrames containing scoring metrics 
        (e.g., mean and standard deviation) for the baseline (non-optimized) models.
        
    scoring_dict_optimizer : dict, optional
        A dictionary where keys are model names and values are DataFrames containing scoring metrics 
        (e.g., mean and standard deviation) for the optimized models. Default is None.
        
    std : bool, optional
        If `True`, returns both the mean and standard deviation of the scores. If `False`, filters the 
        results to only include the mean scores. Default is `False`.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing the combined scoring metrics:
        - If `scoring_dict_optimizer` is provided, the result includes both baseline and optimized scores, 
          with column names indicating the source (e.g., `model_baseline` and `model_optimized`).
        - If `std` is `False`, the result includes only the mean scores. Otherwise, both mean and standard 
          deviation scores are included.
        - If `scoring_dict_optimizer` is not provided, only the baseline scores are returned.

    Example:
    --------
    >>> scoring_dict_trainer = {
    >>>     "model1": pd.DataFrame({"mean": [0.85], "std": [0.03]}),
    >>>     "model2": pd.DataFrame({"mean": [0.80], "std": [0.04]})
    >>> }
    >>> scoring_dict_optimizer = {
    >>>     "model1": pd.DataFrame({"mean": [0.88], "std": [0.02]}),
    >>>     "model2": pd.DataFrame({"mean": [0.83], "std": [0.03]})
    >>> }
    >>> ResultHandler(scoring_dict_trainer, scoring_dict_optimizer, std=False)
    """
    import pandas as pd

    # Input validation
    if not isinstance(scoring_dict_trainer, dict):
        raise ValueError("scoring_dict_trainer must be a dictionary.")

    if scoring_dict_optimizer is not None and not isinstance(scoring_dict_optimizer, dict):
        raise ValueError("scoring_dict_optimizer must be a dictionary if provided.")

    for key, value in scoring_dict_trainer.items():
        if not isinstance(value, pd.DataFrame):
            raise ValueError(f"Value for key '{key}' in scoring_dict_trainer must be a pandas DataFrame.")

    if scoring_dict_optimizer:
        for key, value in scoring_dict_optimizer.items():
            if not isinstance(value, pd.DataFrame):
                raise ValueError(f"Value for key '{key}' in scoring_dict_optimizer must be a pandas DataFrame.")

    if not isinstance(std, bool):
        raise ValueError("std must be a boolean value.")

    # Process scores
    if scoring_dict_optimizer:
        baseline_score = {f"{model_name}_baseline": df for model_name, df in scoring_dict_trainer.items()}
        optimized_score = {f"{model_name}_optimized": df for model_name, df in scoring_dict_optimizer.items()}
        all_scores = {**baseline_score, **optimized_score}

        df = pd.concat({model_name: all_scores[model_name] for model_name in all_scores}, axis=1).sort_index(axis=1)

        # If std == True, skip next step and return both mean and standard deviation of the scores
        if not std:
            # Filter only by the mean score and then drop the 2nd level of column names
            df = df.loc[:, df.columns.get_level_values(1) == "mean"].droplevel(level=1, axis=1)
    else:
        df = pd.concat(scoring_dict_trainer, axis=1)

    return df