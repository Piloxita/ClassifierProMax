def Result_Handler(scoring_dict):
    std_df = pd.concat(scoring_dict, axis='columns').reset_index()
    mean_df = pd.concat(cross_val_results, axis='columns').reset_index()

    print(std_df)
    print(mean_df)

    return mean_df, std_df