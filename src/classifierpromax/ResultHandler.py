import pandas as pd

def ResultHandler(scoring_dict, models=None):
    if not scoring_dict:
        return pd.DataFrame(columns=[], index=['accuracy', 'precision', 'recall', 'f1'])


    index = ['accuracy', 'precision', 'recall', 'f1'] 
    data = {model: list(scores.values()) for model, scores in scoring_dict.items()}

 
    if models:
        for model_name, model in models.items():
            for param, value in model.get_params().items():    
                data.setdefault(model_name, []).append(value)
                index.append(f"{model_name}__{param}")

    num_rows = len(index)
    for model in data:
        if len(data[model]) != num_rows:

            data[model].extend([None] * (num_rows - len(data[model])))


    return pd.DataFrame(data, index=index)
