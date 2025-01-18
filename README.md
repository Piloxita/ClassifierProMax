# classifierpromax

<img src="./img/classifierpromax_logo.png" alt="drawing" width="200"/>

classifierpromax is a scikit-learn wrapper library that helps to train and optimize multiple classifier models in parallel.

`classifier_trainer()`:
This function trains four classifier models using default hyperparameter values. These are used as a baseline for model performance.

`feature_selector()`:
This function will perform feature selection on all input models.

`classifier_optimizer()`:
This function will perform hyperparameter optimization on all input models.

`results_handler()`:
This function will score all the input models based on the criteria specified and visualize the results.

In a machine learning pipeline, code can often be repeated when working with multiple models, violating the DRY (Don’t-Repeat-Yourself) principle. This Python library is to promote DRY principles in machine learning code and create cleaner code.

## Installation

```bash
$ pip install classifierpromax
```

## Usage
1. Training baseline models
```python
import pandas as pd
import numpy as np
from classifierpromax.Classifier_Trainer import Classifier_Trainer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Dummy data
X = pd.DataFrame(np.random.rand(100, 5), columns=[f"feature_{i}" for i in range(5)])
y = pd.Series(np.random.randint(0, 2, size=100))

# Define preprocessor
preprocessor = StandardScaler()

# Function will return a dictionary of models
baseline_models, baseline_score = Classifier_Trainer(preprocessor, X, y, pos_label=1, seed=123)
```
2. Feature selection
```python
from classifierpromax.Feature_Selector import Feature_Selector

# Function will return a dictionary of models
fs_models = Feature_Selector(preprocessor, models, X, y, n_features_to_select=3)
```
3. Hyperparameter optimization
```python
from classifierpromax.Classifier_Optimizer import Classifier_Optimizer

# Function will return a dictionary of optimized models and another dictionary with the scores
opt_models, opt_score = Classifier_Optimizer(fs_models, X, y)
```
4. Results summary
```python
from classifierpromax.Results_Handler import Results_Handler

# Function will score the models and return a summary table
summary = results_handler(opt_score)
```
## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## Contibutors

Long Nguyen, Jenson Chang, Gunisha Kaur, Han Wang

## License

`classifierpromax` was created by Long Nguyen, Jenson Chang, Gunisha Kaur, Han Wang. It is licensed under the terms of the MIT license.

## Credits

`classifierpromax` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
