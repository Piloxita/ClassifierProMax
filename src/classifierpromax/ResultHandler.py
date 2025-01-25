import pandas as pd

def ResultHandler(scoring_dict_trainer, scoring_dict_optimizer):
    """
    Combine results from baseline model training and optimized model tuning into a single DataFrame.
    Exclude index from the optimized models.
    """

    df1 = pd.concat(scoring_dict_trainer.values(), axis='columns').reset_index()
    

    df2 = pd.concat(scoring_dict_optimizer.values(), axis='columns').reset_index()
    

    df2.drop(columns=["index"], inplace=True)
    

    combined_df = pd.concat([df1, df2], axis=1)
  
    combined_df.columns = pd.MultiIndex.from_tuples(
        [(col.split('_')[0], col.split('_')[1]) if len(col.split('_')) > 1 else ('', col) for col in combined_df.columns]
    )
    
    return combined_df

