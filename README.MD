## Discrimination free Naive Bayes

This is a project for a seminar on Fairness-aware machine learning (Autumn 2015)

It aims to implement some of the methods described in the paper Calders10 (see research folder),
introducing ways to make a Naive Bayes classifier non discriminatory

## Code

The majority of the code can be found in bayes.py

Some auxiliary functions for the models can be found in bayes_utils. (some
  credit to  Jason Brownlee -
   http://machinelearningmastery.com/naive-bayes-classifier-scratch-python/ -
   as some of his code was used to reduce boilerplate code)

# Binary Bayes Model

A simple Naive Bayes Model

# Split Fair Bayes Model (2M Model)

This model splits the dataset into ```n``` subsets, one for each of the values
of the sensitive parameter that we do not want to discriminate against.

By creating a model for each of the subset, it minimizes the discrimination
of the model.

# Balanced Bayes Model (MODIFIED BAYES)

This model balances the dataset, making it more fair, tweaking the likelihood by
changing the number of occurrences in the model. It makes it, however, without
disturbing the probability of classifying sample x as the positive class (eg.
  if we talk about loan attribution, we want to keep the total number of
  loans the same as before)

# RESULTS
```
NORMAL MODEL:
Accuracy: 78.4165591794
Discrimination score: 0.398512741018

2M MODEL:
Accuracy: 78.5332596278
Discrimination score: 0.165316892258

MODIFIED MODEL:
Accuracy: 75.9412812481
Discrimination score: -0.0101233590263

```
