# Twitter Sentiment Analysis with Logistic Regression

This project implements a logistic regression model for sentiment analysis on Twitter data, classifying tweets as positive or negative based on word frequencies. The code uses natural language processing (NLP) techniques to process tweets, extract features, train a model, and evaluate its performance.

## Overview

The script processes a dataset of positive and negative tweets from NLTK's `twitter_samples`, extracts features based on word frequencies, and trains a logistic regression model to predict tweet sentiment. It includes functions to preprocess tweets, compute the model's cost, train the model using gradient descent, make predictions, and evaluate accuracy on a test set.

## Requirements

To run the code, you need the following Python libraries:

- `nltk`: For NLP and accessing Twitter sample data.
- `numpy`: For numerical computations.
- `pandas`: For handling data (though minimally used here).
- `utils`: A custom module with helper functions (not included here).

You can install the required libraries using pip:

```bash
pip install nltk numpy pandas