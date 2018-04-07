import string
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphviz
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


# train-validation sets
with open('module-6-assignment-train-idx.json', 'r') as f:
  train_json = json.load(f)
with open('module-6-assignment-validation-idx.json', 'r') as f:
  validation_json = json.load(f)
train_data = loans.iloc[train_json]
validation_data = loans.iloc[validation_json]

#########################################################################################################################


# Decision tree implementation
# Function to count number of mistakes while predicting majority class
def intermediate_node_num_mistakes(labels_in_node):
  if len(labels_in_node) == 0:
    return 0
  safe_loans = (labels_in_node == 1).sum()
  risky_loans = (labels_in_node == -1).sum()
  return min(safe_loans, risky_loans)


# Early stopping condition 2: Minimum node size
def reached_minimum_node_size(data, min_node_size):
    # Return True if the number of data points is less than or equal to the minimum node size.
  if len(data) <= min_node_size:
    return True
  else:
    return False


# Early stopping condition 3: Minimum gain in error reduction
def error_reduction(error_before_split, error_after_split):
  return error_before_split - error_after_split


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


def count_nodes(tre):
  if tre['is_leaf']:
    return 1
  else:
    return 1 + count_nodes(tre['left']) + count_nodes(tre['right'])


def count_leaves(tre):
  if tre['is_leaf']:
    return 1
  return count_leaves(tre['left']) + count_leaves(tre['right'])


def decision_tree_create(data, features, target, current_depth=0, max_depth=10, min_node_size=1, min_error_reduction=0.0):
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

  # Early stopping condition 2: Reached the minimum node size.
  # If the number of data points is less than or equal to the minimum size, return a leaf.
  if reached_minimum_node_size(data, min_node_size):
    print "Early stopping condition 2 reached. Reached minimum node size."
    return create_leaf(target_values)

  # Find the best splitting feature (recall the function best_splitting_feature implemented above)
  splitting_feature = best_splitting_feature(data, features, target)

  # Split on the best feature that we found.
  left_split = data[data[splitting_feature] == 0]
  right_split = data[data[splitting_feature] == 1]

  # Early stopping condition 3: Minimum error reduction
  # Calculate the error before splitting (number of misclassified examples
  # divided by the total number of examples)
  error_before_split = intermediate_node_num_mistakes(target_values) / float(len(data))

  # Calculate the error after splitting (number of misclassified examples
  # in both groups divided by the total number of examples)
  left_mistakes = intermediate_node_num_mistakes(left_split[target])
  right_mistakes = intermediate_node_num_mistakes(right_split[target])
  error_after_split = (left_mistakes + right_mistakes) / float(len(data))

  # If the error reduction is LESS THAN OR EQUAL TO min_error_reduction, return a leaf.
  if error_reduction(error_before_split, error_after_split) <= min_error_reduction:
    print "Early stopping condition 3 reached. Minimum error reduction."
    return create_leaf(target_values)

  remaining_features.remove(splitting_feature)
  print "Split on feature %s. (%s, %s)" % (splitting_feature, len(left_split), len(right_split))

  # Repeat (recurse) on left and right subtrees
  left_tree = decision_tree_create(left_split, remaining_features, target, current_depth + 1, max_depth, min_node_size, min_error_reduction)
  right_tree = decision_tree_create(right_split, remaining_features, target, current_depth + 1, max_depth, min_node_size, min_error_reduction)

  return {'is_leaf': False,
          'prediction': None,
          'splitting_feature': splitting_feature,
          'left': left_tree,
          'right': right_tree}



#  Train a tree model on the train_data
features = list(train_data.columns)
features.remove('safe_loans')


'''
small_decision_tree = decision_tree_create(train_data, features, 'safe_loans', max_depth=2,
                                           min_node_size=10, min_error_reduction=0.0)

if count_nodes(small_decision_tree) == 7:
  print 'Test passed!'
else:
  print 'Test failed... try again!'
  print 'Number of nodes found                :', count_nodes(small_decision_tree)
  print 'Number of nodes that should be there : 7'
'''


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


# Evaluating your decision tree
def evaluate_classification_error(tre, data):
    # Apply the classify(tree, x) to each row in your data
  prediction = data.apply(lambda x: classify(tre, x), axis=1)
  return (data['safe_loans'] != np.array(prediction)).values.sum() * 1. / len(data)


my_decision_tree_new = decision_tree_create(train_data, features, 'safe_loans', max_depth=6,
                                            min_node_size=100, min_error_reduction=0.0)

my_decision_tree_old = decision_tree_create(train_data, features, 'safe_loans', max_depth=6,
                                            min_node_size=0, min_error_reduction=-1)


predict_new = classify(my_decision_tree_new, validation_data.iloc[0], annotate=True)
predict_old = classify(my_decision_tree_old, validation_data.iloc[0], annotate=True)
print(predict_new, predict_old)


new_classification_error = evaluate_classification_error(my_decision_tree_new, validation_data)
old_classification_error = evaluate_classification_error(my_decision_tree_old, validation_data)
print(old_classification_error, new_classification_error)


#########################################################################################################################


# Exploring the effect of max_depth
model_1 = decision_tree_create(train_data, features, 'safe_loans', max_depth=2,
                               min_node_size=0, min_error_reduction=-1)
model_2 = decision_tree_create(train_data, features, 'safe_loans', max_depth=6,
                               min_node_size=0, min_error_reduction=-1)
model_3 = decision_tree_create(train_data, features, 'safe_loans', max_depth=14,
                               min_node_size=0, min_error_reduction=-1)

model_1_train_classification_error = evaluate_classification_error(model_1, train_data)
model_2_train_classification_error = evaluate_classification_error(model_2, train_data)
model_3_train_classification_error = evaluate_classification_error(model_3, train_data)
print(model_1_train_classification_error, model_2_train_classification_error, model_3_train_classification_error)


model_1_validation_classification_error = evaluate_classification_error(model_1, validation_data)
model_2_validation_classification_error = evaluate_classification_error(model_2, validation_data)
model_3_validation_classification_error = evaluate_classification_error(model_3, validation_data)
print(model_1_validation_classification_error, model_2_validation_classification_error, model_3_validation_classification_error)

model_1_complexity = count_leaves(model_1)
model_2_complexity = count_leaves(model_2)
model_3_complexity = count_leaves(model_3)
print(model_1_complexity, model_2_complexity, model_3_complexity)


#########################################################################################################################

# Exploring the effect of min_error

model_4 = decision_tree_create(train_data, features, 'safe_loans', max_depth=6,
                               min_node_size=0, min_error_reduction=-1)
model_5 = decision_tree_create(train_data, features, 'safe_loans', max_depth=6,
                               min_node_size=0, min_error_reduction=0)
model_6 = decision_tree_create(train_data, features, 'safe_loans', max_depth=6,
                               min_node_size=0, min_error_reduction=5)

model_4_validation_classification_error = evaluate_classification_error(model_4, validation_data)
model_5_validation_classification_error = evaluate_classification_error(model_5, validation_data)
model_6_validation_classification_error = evaluate_classification_error(model_6, validation_data)
print(model_4_validation_classification_error, model_5_validation_classification_error, model_6_validation_classification_error)


model_4_complexity = count_leaves(model_4)
model_5_complexity = count_leaves(model_5)
model_6_complexity = count_leaves(model_6)
print(model_4_complexity, model_5_complexity, model_6_complexity)


#########################################################################################################################

# Exploring the effect of min_node_size
model_7 = decision_tree_create(train_data, features, 'safe_loans', max_depth=6,
                               min_node_size=0, min_error_reduction=-1)
model_8 = decision_tree_create(train_data, features, 'safe_loans', max_depth=6,
                               min_node_size=2000, min_error_reduction=-1)
model_9 = decision_tree_create(train_data, features, 'safe_loans', max_depth=6,
                               min_node_size=50000, min_error_reduction=-1)

model_7_validation_classification_error = evaluate_classification_error(model_7, validation_data)
model_8_validation_classification_error = evaluate_classification_error(model_8, validation_data)
model_9_validation_classification_error = evaluate_classification_error(model_9, validation_data)
print(model_7_validation_classification_error, model_8_validation_classification_error, model_9_validation_classification_error)


model_7_complexity = count_leaves(model_7)
model_8_complexity = count_leaves(model_8)
model_9_complexity = count_leaves(model_9)
print(model_7_complexity, model_8_complexity, model_9_complexity)
