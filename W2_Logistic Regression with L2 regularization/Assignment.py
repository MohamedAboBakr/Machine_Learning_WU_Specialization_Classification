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
    return +1
  return -1


def sigmoid(z):
  z = np.exp(-z)
  z = 1 + z
  return 1 / z


features = ['name', 'review', 'rating']
products = pd.read_csv('amazon_baby_subset.csv')
products = products.fillna({'review': ''})
products['review_clean'] = products['review'].apply(remove_punctuation)
products = products[products['rating'] != 3]
products['sentiment'] = products['rating'].apply(get_sentiment)

with open('module-4-assignment-train-idx.json', 'r') as f:
  train_json = json.load(f)
with open('module-4-assignment-validation-idx.json', 'r') as f:
  validation_json = json.load(f)
with open('important_words.json', 'r') as f:
  important_words_json = json.load(f)

train_data = products.iloc[train_json, :]
validation_data = products.iloc[validation_json, :]
important_words = pd.DataFrame({'word': ['(intercept)'] + important_words_json})
vectorizer_word_subset = CountVectorizer(vocabulary=important_words['word'][1:])
train_matrix = vectorizer_word_subset.fit_transform(train_data['review_clean'])
validation_matrix = vectorizer_word_subset.fit_transform(validation_data['review_clean'])

##################################################################################################################

L2_values = [0, 4, 10, 100, 1000, 100000]
models_accuracy_on_training = []
models_accuracy_on_validation = []
models_coeffs = []
train_len = train_data.shape[0]
validation_len = validation_data.shape[0]
top_positive_words = []
top_negative_words = []


# study first case alone when alpha=0 so there is no regularization
model_0 = LogisticRegression()
model_0.fit(train_matrix, train_data['sentiment'])
model_0_coeff = model_0.coef_.flatten().tolist()
model_0_intercept = model_0.intercept_.flatten().tolist()
all_w = np.array(model_0_intercept + model_0_coeff)
important_words['0'] = all_w.reshape((194, 1))

np_coeff = np.array(model_0_coeff)
ind_sorted = np_coeff.argsort()
top_pos_indx = ind_sorted[-5:]
top_neg_indx = ind_sorted[:5]
top_positive_words = [important_words.iloc[i + 1]['word'] for i in top_pos_indx]
top_negative_words = [important_words.iloc[i + 1]['word'] for i in top_neg_indx]
models_coeffs.append(important_words['0'])

train_predict = model_0.predict(train_matrix)
train_diff = train_predict - train_data['sentiment']
train_acc = np.sum(sum(train_diff == 0))
train_accuracy = 1. * train_acc / train_len
models_accuracy_on_training.append(train_accuracy)

validation_predict = model_0.predict(validation_matrix)
validation_diff = validation_predict - validation_data['sentiment']
validation_acc = np.sum(sum(validation_diff == 0))
validation_accuracy = 1. * validation_acc / validation_len
models_accuracy_on_validation.append(validation_accuracy)

##################################################################################################################

# study the rest of alphas
for i in range(1, len(L2_values)):
  alpha = L2_values[i]
  C = 1.0 / alpha
  model = LogisticRegression(C=C, penalty='l2')
  model.fit(train_matrix, train_data['sentiment'])
  model_coeff = model.coef_.flatten().tolist()
  model_intercept = model.intercept_.flatten().tolist()
  all_w = np.array(model_intercept + model_coeff)
  important_words[str(alpha)] = all_w.reshape((194, 1))
  models_coeffs.append(important_words[str(alpha)])

  train_predict = model.predict(train_matrix)
  train_diff = train_predict - train_data['sentiment']
  train_acc = np.sum(sum(train_diff == 0))
  train_accuracy = 1. * train_acc / train_len
  models_accuracy_on_training.append(train_accuracy)

  validation_predict = model.predict(validation_matrix)
  validation_diff = validation_predict - validation_data['sentiment']
  validation_acc = np.sum(sum(validation_diff == 0))
  validation_accuracy = 1. * validation_acc / validation_len
  models_accuracy_on_validation.append(validation_accuracy)


for i in range(0, len(L2_values)):
  print(L2_values[i], '   ', models_accuracy_on_training[i], '   ', models_accuracy_on_validation[i])


##################################################################################################################


plt.rcParams['figure.figsize'] = 10, 6


def make_coefficient_plot(table, positive_words, negative_words, l2_penalty_list):
  cmap_positive = plt.get_cmap('Reds')
  cmap_negative = plt.get_cmap('Blues')

  xx = l2_penalty_list
  plt.plot(xx, [0.] * len(xx), '--', lw=1, color='k')

  table_positive_words = table[table['word'].isin(positive_words)]
  table_negative_words = table[table['word'].isin(negative_words)]
  print(table_positive_words)
  print(table_negative_words)
  del table_positive_words['word']
  del table_negative_words['word']

  for i in xrange(len(positive_words)):
    color = cmap_positive(0.8 * ((i + 1) / (len(positive_words) * 1.2) + 0.15))
    print(table_positive_words[i:i + 1].as_matrix().flatten())
    plt.plot(xx, table_positive_words[i:i + 1].as_matrix().flatten(),
             '-', label=positive_words[i], linewidth=4.0, color=color)

  for i in xrange(len(negative_words)):
    color = cmap_negative(0.8 * ((i + 1) / (len(negative_words) * 1.2) + 0.15))
    plt.plot(xx, table_negative_words[i:i + 1].as_matrix().flatten(),
             '-', label=negative_words[i], linewidth=4.0, color=color)

  plt.legend(loc='best', ncol=3, prop={'size': 16}, columnspacing=0.5)
  plt.axis([1, 1e5, -3, 3])
  plt.title('Coefficient path')
  plt.xlabel('L2 penalty ($\lambda$)')
  plt.ylabel('Coefficient value')
  plt.xscale('log')
  plt.rcParams.update({'font.size': 18})
  plt.tight_layout()
  plt.show()


make_coefficient_plot(important_words, top_positive_words, top_negative_words, L2_values)
