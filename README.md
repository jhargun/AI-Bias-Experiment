# AI-Bias-Experiment
This repository contains code for the final project of CS492 (Social Implications of Computing). It contains code for preprocessing IPUMS data and using this preprocessed data to train/evaluate a set of machine learning models.

The results of the machine learning models can be found at this [website](https://cs492-project-woad.vercel.app/). The code for the [frontend](https://github.com/vicswu/CS492-Project) and [backend](https://github.com/shaishav-p/cs492-inference-service) can be found in 2 separate Github repositories.

## Preprocessing data

To preprocess the data, you will first need the raw data provided by [IPUMS](https://usa.ipums.org/usa-action/), which will be in a .csv file. First, use the [data_preprocess.ipynb](https://github.com/jhargun/AI-Bias-Experiment/blob/main/models/data_preprocess.ipynb) notebook to do some initial preprocessing. Next, use the [large_field_preprocessing.ipynb](https://github.com/jhargun/AI-Bias-Experiment/blob/main/models/large_field_preprocessing.ipynb) notebook to preprocess additional fields. These 2 notebooks are separate because 3 fields (state, degree, and occupation) require a disproportionate number of categorical columns to represent. Therefore, it can be useful to have the preprocessed files from the initial notebook, which will be far smaller than the final preprocessed files.

Note that data analysis for the dataset can be done using the [data_analysis.ipynb](https://github.com/jhargun/AI-Bias-Experiment/blob/main/models/data_analysis.ipynb) notebook. This can help you identify trends and potential sources of biases in your data. If you don't want to rerun custom analysis, this analysis is provided in the [misc_data folder](https://github.com/jhargun/AI-Bias-Experiment/tree/main/models/misc_data).

## Training models

To train models, use the [training.ipynb](https://github.com/jhargun/AI-Bias-Experiment/blob/main/models/training/training.ipynb) notebook. Note that the final result only used regression models. Classification models and random forests were considered, and are therefore present in different files, but were not used.

## Checking for Bias

Code to run a SHAP analysis is in [sklearn_training.ipynb](https://github.com/jhargun/AI-Bias-Experiment/blob/main/models/training/sklearn_training.ipynb). This can be used to give a general sense of potential sources of bias. For more in depth metrics, use the [bias_analysis.ipynb](https://github.com/jhargun/AI-Bias-Experiment/blob/main/models/bias_analysis.ipynb) notebook.
