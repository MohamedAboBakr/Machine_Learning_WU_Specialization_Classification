import string
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import log, sqrt

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm


def remove_punctuation(text):
  return text.translate(None, string.punctuation)


def get_sentiment(rating):
  if rating > 3:
    return 1
  return -1


def sigmoid(z):
  z = np.exp(-z)
  z = 1 + z
  return 1 / z


features = ['name', 'review', 'rating']
products = pd.read_csv('amazon_baby.csv')
products = products.fillna({'review': ''})
products['review_clean'] = products['review'].apply(remove_punctuation)
products = products[products['rating'] != 3]
products['sentiment'] = products['rating'].apply(get_sentiment)


with open('module-2-assignment-train-idx.json', 'r') as f:
  train_json = json.load(f)
with open('module-2-assignment-test-idx.json', 'r') as f:
  test_json = json.load(f)

train_data = products.iloc[train_json, :]
test_data = products.iloc[test_json, :]


class_1 = test_data[test_data['sentiment'] == 1].shape[0]
class_0 = test_data.shape[0] - class_1
accuracy_test_data_majority_class_classifier_model = 1.0 * max(class_1, class_0) / test_data.shape[0]
print(accuracy_test_data_majority_class_classifier_model)

#########################################################################################################################


# MODEL_1
vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
train_matrix = vectorizer.fit_transform(train_data['review_clean'])
test_matrix = vectorizer.transform(test_data['review_clean'])

model = LogisticRegression()
model.fit(train_matrix, train_data['sentiment'])
print(model.coef_)
print (np.sum(sum(model.coef_ >= 0)))

# testing model on small sample from testing dataset
sample_test_data = test_data[10:13]
sample_test_matrix = vectorizer.transform(sample_test_data['review_clean'])
scores = model.decision_function(sample_test_matrix)
print(scores)

predict1 = []
predict3 = []
for s in scores:
  if s > 0:
    predict1.append(1)
  else:
    predict1.append(-1)

  predict3.append(sigmoid(s))

predict2 = model.predict(sample_test_matrix)

print(predict1)
print(predict2)
print(predict3)


# testing model on all testing dataset
scores = model.decision_function(test_matrix)
predictions = sigmoid(scores)
test_data['predictions'] = predictions
print(test_data.sort_values('predictions', ascending=True).iloc[0:20])
print(test_data.sort_values('predictions', ascending=False).iloc[0:20])

# accuracy
predict_train = model.predict(train_matrix)
train_diff = predict_train - train_data['sentiment']
train_acc = np.sum(sum(train_diff == 0))
total_train = len(predict_train)
train_accuracy = 1. * train_acc / total_train

predict_test = model.predict(test_matrix)
test_diff = predict_test - test_data['sentiment']
test_acc = np.sum(sum(test_diff == 0))
total_test = len(predict_test)
test_accuracy = 1. * test_acc / total_test

print(train_accuracy, test_accuracy)

#########################################################################################################################
# MOdel 2
# Learn another classifier with fewer words

significant_words = ['love', 'great', 'easy', 'old', 'little', 'perfect', 'loves',
                     'well', 'able', 'car', 'broke', 'less', 'even', 'waste', 'disappointed',
                     'work', 'product', 'money', 'would', 'return']
vectorizer_word_subset = CountVectorizer(vocabulary=significant_words)
train_matrix_word_subset = vectorizer_word_subset.fit_transform(train_data['review_clean'])
test_matrix_word_subset = vectorizer_word_subset.transform(test_data['review_clean'])


simple_model = LogisticRegression()
simple_model.fit(train_matrix_word_subset, train_data['sentiment'])
coeff = simple_model.coef_.flatten()
coeff_word = []
for coef, word in zip(coeff, significant_words):
  coeff_word.append((coef, word))
coeff_word.sort()
print(coeff_word)


# accuracy
predict_train = simple_model.predict(train_matrix_word_subset)
train_diff = predict_train - train_data['sentiment']
train_acc = np.sum(sum(train_diff == 0))
total_train = len(predict_train)
train_accuracy = 1. * train_acc / total_train

predict_test = simple_model.predict(test_matrix_word_subset)
test_diff = predict_test - test_data['sentiment']
test_acc = np.sum(sum(test_diff == 0))
total_test = len(predict_test)
test_accuracy = 1. * test_acc / total_test

print(train_accuracy, test_accuracy)
