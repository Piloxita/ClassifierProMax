# classifierpromax

<img src="https://github.com/UBC-MDS/ClassifierProMax/blob/75d4f39c2e75ceff955005e6d443be4151ecc40a/img/classifierpromax_logo.png?raw=true" alt="drawing" width="200"/>

[![Documentation Status](https://readthedocs.org/projects/classifierpromax/badge/?version=latest)](https://classifierpromax.readthedocs.io/en/latest/?badge=latest)

`classifierpromax` is a scikit-learn wrapper library that helps to train and optimize multiple classifier models in parallel.

`ClassifierTrainer()`:
This function trains three classifier models using default hyperparameter values. These are used as a baseline for model performance.

`FeatureSelector()`:
This function will perform feature selection on all input models.

`ClassifierOptimizer()`:
This function will perform hyperparameter optimization on all input models.

`ResultsHandler()`:
This function will return the score and hyperparameters for all the input models.

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
from classifierpromax.ClassifierTrainer import ClassifierTrainer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Dummy data
X = pd.DataFrame(np.random.rand(100, 5), columns=[f"feature_{i}" for i in range(5)])
y = pd.Series(np.random.randint(0, 2, size=100))

# Define preprocessor
preprocessor = StandardScaler()

# Function will return a dictionary of models
baseline_models, baseline_score = ClassifierTrainer(preprocessor, X, y, seed=123)
```
2. Feature selection
```python
from classifierpromax.FeatureSelector import FeatureSelector

# Function will return a dictionary of models
fs_models = FeatureSelector(preprocessor, models, X, y, n_features_to_select=3)
```
3. Hyperparameter optimization
```python
from classifierpromax.ClassifierOptimizer import ClassifierOptimizer

# Function will return a dictionary of optimized models and another dictionary with the scores
opt_models, opt_score = ClassifierOptimizer(fs_models, X, y, scoring="f1")
```
4. Results summary
```python
from classifierpromax.ResultHandler import ResultHandler

# Function will score the models and return a summary table
summary = ResultHandler(score, opt_models)
print(summary)
```
## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## Contributors

Long Nguyen, Jenson Chang, Gunisha Kaur, Han Wang

## License

`classifierpromax` was created by Long Nguyen, Jenson Chang, Gunisha Kaur, Han Wang. It is licensed under the terms of the MIT license.

## Credits

`classifierpromax` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
