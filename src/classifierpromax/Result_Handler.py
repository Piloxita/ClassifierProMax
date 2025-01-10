def Result_Handler(scoring_dict):
    """
    Handles the results of cross-validation scoring by concatenating the input dictionary of scoring data 
    into two DataFrames: one for the mean and one for the standard deviation. The function prints these DataFrames 
    and returns them as a tuple.

    Parameters:
    ----------
    scoring_dict (dict): A dictionary containing the scoring results, where the keys are column names 
                          and the values are the corresponding scoring data for each fold of cross-validation.

    Returns:
    -------
    tuple: A tuple containing two DataFrames:
           - mean_df: DataFrame containing the mean of the scores.
           - std_df: DataFrame containing the standard deviation of the scores.
    
    Example:
    --------
    # Example of scoring_dict containing the results for 5 folds
    scoring_dict = {
        'mean': {'fold_1': [0.85], 'fold_2': [0.87], 'fold_3': [0.86], 'fold_4': [0.84], 'fold_5': [0.88]},
        'std': {'fold_1': [0.02], 'fold_2': [0.01], 'fold_3': [0.02], 'fold_4': [0.01], 'fold_5': [0.01]}
    }

    mean_df, std_df = Result_Handler(scoring_dict)
    print(mean_df)
    print(std_df)
    """

    # # Assuming that 'scoring_dict' contains two parts: mean and standard deviation
    std_df = pd.concat(scoring_dict['std'], axis='columns').reset_index()
    mean_df = pd.concat(scoring_dict['mean'], axis='columns').reset_index()

    print(std_df)
    print(mean_df)

    return mean_df, std_df