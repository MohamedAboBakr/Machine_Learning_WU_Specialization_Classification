import string
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphviz
from sklearn import tree
from os import system
from math import log, sqrt
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier

# load data set
loans = pd.read_csv('lending-club-data.csv')
loans['safe_loans'] = loans['bad_loans'].apply(lambda x: +1 if x == 0 else -1)
del loans['bad_loans']

features = ['grade',              # grade of the loan
            'term',               # the term of the loan
            'home_ownership',     # home_ownership status: own, mortgage or rent
            'emp_length',         # number of years of employment
            ]
target = 'safe_loans'
loans = loans[features + [target]]


# one-hot encoding
categorical_variables = []
for feat_name, feat_type in zip(loans.columns, loans.dtypes):
  if feat_type == object:
    categorical_variables.append(feat_name)

for feature in categorical_variables:

  loans_one_hot_encoded = pd.get_dummies(loans[feature], prefix=feature)
  # print loans_one_hot_encoded

  loans = loans.drop(feature, axis=1)
  for col in loans_one_hot_encoded.columns:
    loans[col] = loans_one_hot_encoded[col]


# train-test sets
with open('module-5-assignment-2-train-idx.json', 'r') as f:
  train_json = json.load(f)
with open('module-5-assignment-2-test-idx.json', 'r') as f:
  test_json = json.load(f)
train_data = loans.iloc[train_json]
test_data = loans.iloc[test_json]

#########################################################################################################################

# Decision tree implementation
# Function to count number of mistakes while predicting majority class


def intermediate_node_num_mistakes(labels_in_node):
  if len(labels_in_node) == 0:
    return 0
  safe_loans = (labels_in_node == 1).sum()
  risky_loans = (labels_in_node == -1).sum()
  return min(safe_loans, risky_loans)

# Function to pick best feature to split on


def best_splitting_feature(data, features, target):
  best_feature = None
  best_error = 10
  num_data_points = float(len(data))
  for feature in features:
    left_split = data[data[feature] == 0]
    right_split = data[data[feature] == 1]
    left_mistakes = intermediate_node_num_mistakes(left_split[target])
    right_mistakes = intermediate_node_num_mistakes(right_split[target])
    error = (left_mistakes + right_mistakes) / num_data_points
    if error < best_error:
      best_error = error
      best_feature = feature
  return best_feature

# Building the tree


def create_leaf(target_values):
  leaf = {'splitting_feature': None,
          'left': None,
          'right': None,
          'is_leaf': True}
  num_ones = len(target_values[target_values == +1])
  num_minus_ones = len(target_values[target_values == -1])
  if num_ones > num_minus_ones:
    leaf['prediction'] = 1
  else:
    leaf['prediction'] = -1
  return leaf


def decision_tree_create(data, features, target, current_depth=0, max_depth=10):
  remaining_features = features[:]  # Make a copy of the features.
  target_values = data[target]
  print "--------------------------------------------------------------------"
  print "Subtree, depth = %s (%s data points)." % (current_depth, len(target_values))
  # Stopping condition 1
  # (Check if there are mistakes at current node.
  # Recall you wrote a function intermediate_node_num_mistakes to compute this.)
  if intermediate_node_num_mistakes(target_values) == 0:
    print "Stopping condition 1 reached."
    # If not mistakes at current node, make current node a leaf node
    return create_leaf(target_values)

  # Stopping condition 2 (check if there are remaining features to consider splitting on)
  if remaining_features == []:
    print "Stopping condition 2 reached."
    # If there are no remaining features to consider, make current node a leaf node
    return create_leaf(target_values)

  # Additional stopping condition (limit tree depth)
  if current_depth >= max_depth:
    print "Reached maximum depth. Stopping for now."
    # If the max tree depth has been reached, make current node a leaf node
    return create_leaf(target_values)

  # Find the best splitting feature (recall the function best_splitting_feature implemented above)
  splitting_feature = best_splitting_feature(data, remaining_features, target)

  # Split on the best feature that we found.
  left_split = data[data[splitting_feature] == 0]
  right_split = data[data[splitting_feature] == 1]
  remaining_features.remove(splitting_feature)
  print "Split on feature %s. (%s, %s)" % (splitting_feature, len(left_split), len(right_split))

  # Create a leaf node if the split is "perfect"
  if len(left_split) == len(data):
    print "Creating leaf node."
    return create_leaf(left_split[target])
  if len(right_split) == len(data):
    print "Creating leaf node."
    return create_leaf(right_split[target])
  # Repeat (recurse) on left and right subtrees
  left_tree = decision_tree_create(left_split, remaining_features, target, current_depth + 1, max_depth)
  right_tree = decision_tree_create(right_split, remaining_features, target, current_depth + 1, max_depth)

  return {'is_leaf': False,
          'prediction': None,
          'splitting_feature': splitting_feature,
          'left': left_tree,
          'right': right_tree}



#  Train a tree model on the train_data
a = list(train_data.columns)
a.remove('safe_loans')
my_decision_tree = decision_tree_create(train_data, a, 'safe_loans', current_depth=0, max_depth=6)


# Making predictions with a decision tree
def classify(tre, x, annotate=False):
  # if the node is a leaf node
  if tre['is_leaf']:
    if annotate:
      print "At leaf, predicting %s" % tre['prediction']
    return tre['prediction']
  else:
    # split on feature.
    split_feature_value = x[tre['splitting_feature']]
    if annotate:
      print "Split on %s = %s" % (tre['splitting_feature'], split_feature_value)
    if split_feature_value == 0:
      return classify(tre['left'], x, annotate)
    else:
      return classify(tre['right'], x, annotate)


# test decision tree
predicted_true = test_data.iloc[0]['safe_loans']
predicted_DT = classify(my_decision_tree, test_data.iloc[0])
print(predicted_true, predicted_DT)

# Evaluating your decision tree


def evaluate_classification_error(tre, data):
    # Apply the classify(tree, x) to each row in your data
  prediction = data.apply(lambda x: classify(tre, x), axis=1)
  return (data['safe_loans'] != np.array(prediction)).values.sum() * 1. / len(data)


classification_error_on_test_data = evaluate_classification_error(my_decision_tree, test_data)
print(classification_error_on_test_data)
