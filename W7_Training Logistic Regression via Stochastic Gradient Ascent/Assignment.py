import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt


def remove_punctuation(text):
  import string
  return text.translate(None, string.punctuation)


def get_numpy_data(dataframe, features, label):
  dataframe['constant'] = 1
  features = ['constant'] + features
  features_frame = dataframe[features]
  feature_matrix = features_frame.as_matrix()
  label_sarray = dataframe[label]
  label_array = label_sarray.as_matrix()
  return(feature_matrix, label_array)


def predict_probability(feature_matrix, coefficients):
  # Take dot product of feature_matrix and coefficients
  score = np.dot(feature_matrix, coefficients)
  # Compute P(y_i = +1 | x_i, w) using the link function
  predictions = 1.0 / (1 + np.exp(-score))
  # return predictions
  return predictions


def feature_derivative(errors, feature):
  # Compute the dot product of errors and feature
  derivative = np.dot(np.transpose(errors), feature)
  # Return the derivative
  return derivative


def compute_avg_log_likelihood(feature_matrix, sentiment, coefficients):

  indicator = (sentiment == +1)
  scores = np.dot(feature_matrix, coefficients)
  logexp = np.log(1. + np.exp(-scores))
  # scores.shape (53072L, 1L)
  # indicator.shape (53072L,)

  # Simple check to prevent overflow
  mask = np.isinf(logexp)
  logexp[mask] = -scores[mask]

  lp = np.sum((indicator.reshape(scores.shape) - 1) * scores - logexp) / len(feature_matrix)

  return lp


# Load and process review dataset
products = pd.read_csv('amazon_baby_subset.csv')
with open('important_words.json') as important_words_file:
  important_words = json.load(important_words_file)

products = products.fillna({'review': ''})
products['review_clean'] = products['review'].apply(remove_punctuation)

for word in important_words:
  products[word] = products['review_clean'].apply(lambda s: s.split().count(word))

# Split data into training and validation sets
with open('module-10-assignment-train-idx.json') as train_data_file:
  train_data_idx = json.load(train_data_file)
with open('module-10-assignment-validation-idx.json') as validation_data_file:
  validation_data_idx = json.load(validation_data_file)

  train_data = products.iloc[train_data_idx]
  validation_data = products.iloc[validation_data_idx]


# Convert DataFrame to NumPy array
feature_matrix_train, sentiment_train = get_numpy_data(train_data, important_words, 'sentiment')
feature_matrix_valid, sentiment_valid = get_numpy_data(validation_data, important_words, 'sentiment')


# Modifying the derivative for stochastic gradient ascent

j = 1                        # Feature number
i = 10                       # Data point number
coefficients = np.zeros(194)  # A point w at which we are computing the gradient.

predictions = predict_probability(feature_matrix_train[i:i + 1, :], coefficients)
indicator = (sentiment_train[i:i + 1] == +1)

errors = indicator - predictions
gradient_single_data_point = feature_derivative(errors, feature_matrix_train[i:i + 1, j])
print ("Gradient single data point: %s" % gradient_single_data_point)
print ("           --> Should print 0.0")


# Modifying the derivative for using a batch of data points
j = 1                        # Feature number
i = 10                       # Data point start
B = 10                       # Mini-batch size
coefficients = np.zeros(194)  # A point w at which we are computing the gradient.

predictions = predict_probability(feature_matrix_train[i:i + B, :], coefficients)
indicator = (sentiment_train[i:i + B] == +1)

errors = indicator - predictions
gradient_mini_batch = feature_derivative(errors, feature_matrix_train[i:i + B, j])
print ("Gradient mini-batch data points: %s" % gradient_mini_batch)
print ("                --> Should print 1.0")
