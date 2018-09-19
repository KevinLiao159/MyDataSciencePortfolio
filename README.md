# Klearn: Data Science and Machine Learning Tool Kits for Kagglers

![Klearn logo](https://github.com/KevinLiao159/klearn/blob/master/images/Klearn-logo.png)

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/KevinLiao159/klearn/blob/master/LICENSE)

## Good Job!!! I am glad that you just found Klearn.
Klearn is a Python module that speeds up data science or machine learning research work flow tremendously. It embraces the best data science practices and commits to empower data scientists. It holds several data science most-use modules, which includes but not limit to EDA module, feature engineering module, cross-validation strategy, hold-out data scoring, and model ensembling.

Klearn is compatible with: __Python 2.7-3.6__.


------------------


## Some principles

- __User friendliness.__ Klearn is designed for data science beginners. Klearn follows best practices for reducing cognitive load: it offers consistent & simple APIs, it minimizes the number of user actions required for common use cases, and it provides clear and actionable feedback upon user error.

- __Modularity.__ A data science research project is understood as a sequence of tasks including EDA, feature engineering, and model selection/benchmarking. Each module in Klearn is reponsible for each task in data scientist research routine work flow.

- __Easy extensibility.__ New modules are simple to add (as new classes and functions), and existing modules provide ample examples. To be able to easily create new modules allows for total expressiveness, making Klearn suitable for advanced research.

- __Work with Python__. No separate models configuration files in a declarative format. Models are described in Python code, which is compact, easier to debug, and allows for ease of extensibility.


------------------


## Module structure

The main modules of Klearn API are:
* `datasets`, which is responsible for dumping data in certain format
* `eda`, which is responsible for data visualization and exploratory analysis
* `ensemble`, which is reponsible for combining models together
* `model_selection`, which holds cv strategy classes and scoring functions
* `models`, which is for higher level wrappers of machine learning models
* `preprocessing`, which responsible for data cleaning and feature engineering

The complete file-structure for the project is as follows:
```
klearn/
    klearn/
        datasets/
            libffm_format.py
        eda/
            eda.py
            plotly.py
            seaborn.py
        ensemble/
            dispatch.py
            ensemble.py
        model_selection/
            metrics.py
            scorers.py
            split.py
        models/
            modifiers.py
            trainers.py
            transformers.py
        preprocessing/
            cleaners.py
            features.py
            targets.py
        logger.py
        utils.py
    images/
        ...random stuff

    README.md
    LICENSE
    requirements.txt
    setup.py
```


------------------


## Installation

- **Install Klearn from PyPI (NOT supported for now):**

```sh
sudo pip install klearn
```

- **Alternatively: install Klearn from the GitHub source (recommended):**

First, clone Klearn using `git`:

```sh
git clone https://github.com/KevinLiao159/klearn.git
```

 Then, `cd` to the Klearn folder and run the install command:
```sh
cd klearn
sudo python setup.py install
```
