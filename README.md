# classifierpromax

Help you train and optimize many popuplar classifiers in one place with a nice result table

## Installation

```bash
$ pip install classifierpromax
```

## Usage

<<<<<<< Updated upstream
- TODO
=======
1. Training baseline models
```python
from classifierpromax import classifier_trainer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = train_test_split(data, test_size=0.2)

# Function will return a dictionary of models
baseline_models, baseline_score = classifier_trainer(StandardScaler(), X, y, pos_label=1, seed=123)
```
2. Feature selection
```python
from classifierpromax import feature_selector

# Function will return a dictionary of models
fs_models= feature_selector(baseline_models, 'RFE')
```
3. Hyperparameter optimization
```python
from classifierpromax import classifier_optimizer

score = "f1"

# Function will return a dictionary of models
opt_models, opt_score = classifier_optimizer(fs_models, X, y, score)
```
4. Results summary
```python
from classifierpromax import results_handler

# Function will score the models and return a summary table
summary = results_handler(opt_score)
```
>>>>>>> Stashed changes

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## Contibutors

Long Nguyen, Jenson Chang, Gunisha Kaur, Han Wang

## License

`classifierpromax` was created by Long Nguyen, Jenson Chang, Gunisha Kaur, Han Wang. It is licensed under the terms of the MIT license.

## Credits

`classifierpromax` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
