# classifierpromax

<img src="https://github.com/UBC-MDS/ClassifierProMax/blob/75d4f39c2e75ceff955005e6d443be4151ecc40a/img/classifierpromax_logo.png?raw=true" alt="drawing" width="200"/>

[![Documentation Status](https://readthedocs.org/projects/classifierpromax/badge/?version=latest)](https://classifierpromax.readthedocs.io/en/latest/?badge=latest)[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/) ![ci-cd](https://github.com/UBC-MDS/classifierpromax/actions/workflows/ci-cd.yml/badge.svg) [![codecov](https://codecov.io/gh/UBC-MDS/classifierpromax/branch/main/graph/badge.svg)](https://codecov.io/gh/UBC-MDS/classifierpromax)

`classifierpromax` is a scikit-learn wrapper library that helps to train and optimize multiple classifier models in parallel.

`ClassifierTrainer()`:
Train multiple machine learning classifiers using cross-validation and return trained models and evaluation metrics.

`FeatureSelector()`:
Selects features for multiple classification models using RFE or Pearson methods.

`ClassifierOptimizer()`:
Optimizes a dictionary of scikit-learn Pipeline classifiers using RandomizedSearchCV and evaluates their performance.

`ResultsHandler()`:
Processes and combines scoring results from model training and optimization.

In a machine learning pipeline, code can often be repeated when working with multiple models, violating the DRY (Don’t-Repeat-Yourself) principle. This Python library is to promote DRY principles in machine learning code and create cleaner code.

## Installation

Before installation, please make sure Python 3.12 or newer is installed. 

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

preprocessor = StandardScaler()
baseline_models, baseline_score = ClassifierTrainer(preprocessor, X, y, seed=123)
```
2. Feature selection
```python
from classifierpromax.FeatureSelector import FeatureSelector

fs_models = FeatureSelector(preprocessor, baseline_models, X, y, n_features_to_select=3)
```
3. Hyperparameter optimization
```python
from classifierpromax.ClassifierOptimizer import ClassifierOptimizer

opt_models, opt_score = ClassifierOptimizer(fs_models, X, y, scoring="f1")
```
4. Results summary
```python
from classifierpromax.ResultHandler import ResultHandler

summary = ResultHandler(baseline_score, opt_score)
print(summary)
```
## Testing
Create a new environment with Python 3.12. 
```bash
conda create -n classifierpromax python=3.12
conda activate classifierpromax
```

Clone the repo.
```bash
git clone git@github.com:UBC-MDS/ClassifierProMax.git
```

Install poetry following these [instructions](https://python-poetry.org/docs/#installing-with-the-official-installer) and then run the following bash command. 
```bash
$ poetry install
```

Execute pytest from the root project directory. 
```bash
$ pytest
```

## Contributing

Interested in contributing? Check out the [contributing guidelines](https://github.com/UBC-MDS/ClassifierProMax/blob/main/CONTRIBUTING.md). Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## Contributors

Long Nguyen, Jenson Chang, Gunisha Kaur, Han Wang

## License

`classifierpromax` was created by Long Nguyen, Jenson Chang, Gunisha Kaur, Han Wang. It is licensed under the terms of the [MIT license](https://github.com/UBC-MDS/ClassifierProMax/blob/main/LICENSE).

## Credits

`classifierpromax` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
