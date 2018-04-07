import string
import json
import random
from math import log
from math import exp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphviz
from os import system
from math import log, sqrt
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier



# load data set
loans = pd.read_csv('lending-club-data2.csv')
loans['safe_loans'] = loans['bad_loans'].apply(lambda x: +1 if x == 0 else -1)
del loans['bad_loans']

features = ['grade',              # grade of the loan
            'term',               # the term of the loan
            'home_ownership',     # home ownership status: own, mortgage or rent
            'emp_length',         # number of years of employment
            ]

target = 'safe_loans'
loans = loans[[target] + features]


# one-hot encoding
categorical_variables = []
for feat_name, feat_type in zip(loans.columns, loans.dtypes):
  if feat_type == object:
    categorical_variables.append(feat_name)

for feature in categorical_variables:
  loans_one_hot_encoded = pd.get_dummies(loans[feature], prefix=feature)
  loans_one_hot_encoded.fillna(0)
  loans = loans.drop(feature, axis=1)
  for col in loans_one_hot_encoded.columns:
    loans[col] = loans_one_hot_encoded[col]


# train-test sets
with open('module-8-assignment-2-train-idx.json', 'r') as f:
  train_json = json.load(f)
with open('module-8-assignment-2-test-idx.json', 'r') as f:
  test_json = json.load(f)


train_data = loans.iloc[train_json]
test_data = loans.iloc[test_json]

features = list(train_data.columns)
features.remove('safe_loans')

#######################################################################################################################


# Weighted error definition
def intermediate_node_weighted_mistakes(labels_in_node, data_weights):
    # Sum the weights of all entries with label +1
  labels_in_node = np.array(labels_in_node)
  data_weights = np.array(data_weights)
  total_weight_positive = np.sum(data_weights[labels_in_node == +1])

  # Weight of mistakes for predicting all -1's is equal to the sum above
  weighted_mistakes_all_negative = total_weight_positive

  # Sum the weights of all entries with label -1

  total_weight_negative = np.sum(data_weights[labels_in_node == -1])

  # Weight of mistakes for predicting all +1's is equal to the sum above
  weighted_mistakes_all_positive = total_weight_negative

  # Return the tuple (weight, class_label) representing the lower of the two weights
  #    class_label should be an integer of value +1 or -1.
  # If the two weights are identical, return (weighted_mistakes_all_positive,+1)
  if weighted_mistakes_all_negative < weighted_mistakes_all_positive:
    return (weighted_mistakes_all_negative, -1)
  else:
    return (weighted_mistakes_all_positive, +1)


# Function to pick best feature to split on
def best_splitting_feature(data, features, target, data_weights):
  best_feature = None
  best_error = float('+inf')
  data['data_weights'] = data_weights

  for feature in features:
    left_split = data[data[feature] == 0]
    right_split = data[data[feature] == 1]

    left_data_weights = left_split['data_weights']
    right_data_weights = right_split['data_weights']

    left_weights_mistakes, left_class = intermediate_node_weighted_mistakes(left_split[target], left_data_weights)
    right_weights_mistakes, right_class = intermediate_node_weighted_mistakes(right_split[target], right_data_weights)

    error = (left_weights_mistakes + right_weights_mistakes) * 1. / sum(data_weights)
    if error < best_error:
      best_error = error
      best_feature = feature
  return best_feature


def create_leaf(target_values, data_weights):

  # Create a leaf node
  leaf = {'splitting_feature': None,
          'is_leaf': True}

  # Computed weight of mistakes.
  weighted_error, best_class = intermediate_node_weighted_mistakes(target_values, data_weights)
  # Store the predicted class (1 or -1) in leaf['prediction']
  leaf['prediction'] = best_class

  return leaf


def weighted_decision_tree_create(data, features, target, data_weights, current_depth=1, max_depth=10):
  remaining_features = features[:]  # Make a copy of the features.
  target_values = data[target]

  data['data_weights'] = data_weights

  print "--------------------------------------------------------------------"
  print "Subtree, depth = %s (%s data points)." % (current_depth, len(target_values))

  # Stopping condition 1. Error is 0.
  if intermediate_node_weighted_mistakes(target_values, data_weights)[0] <= 1e-15:
    print "Stopping condition 1 reached."
    return create_leaf(target_values, data_weights)

  # Stopping condition 2. No more features.
  if remaining_features == []:
    print "Stopping condition 2 reached."
    return create_leaf(target_values, data_weights)

  # Additional stopping condition (limit tree depth)
  if current_depth > max_depth:
    print "Reached maximum depth. Stopping for now."
    return create_leaf(target_values, data_weights)

  # If all the datapoints are the same, splitting_feature will be None. Create a leaf
  splitting_feature = best_splitting_feature(data, features, target, data_weights)
  remaining_features.remove(splitting_feature)

  left_split = data[data[splitting_feature] == 0]
  right_split = data[data[splitting_feature] == 1]

  """
    left_data_weights = data_weights[data[splitting_feature] == 0]
    right_data_weights = data_weights[data[splitting_feature] == 1]
    """
  left_data_weights = np.array(left_split['data_weights'])
  right_data_weights = np.array(right_split['data_weights'])

  print "Split on feature %s. (%s, %s)" % (
      splitting_feature, len(left_split), len(right_split))

  # Create a leaf node if the split is "perfect"
  if len(left_split) == len(data):
    print "Creating leaf node."
    return create_leaf(left_split[target], data_weights)
  if len(right_split) == len(data):
    print "Creating leaf node."
    return create_leaf(right_split[target], data_weights)

  # Repeat (recurse) on left and right subtrees
  left_tree = weighted_decision_tree_create(
      left_split, remaining_features, target, left_data_weights, current_depth + 1, max_depth)
  right_tree = weighted_decision_tree_create(
      right_split, remaining_features, target, right_data_weights, current_depth + 1, max_depth)

  return {'is_leaf': False,
          'prediction': None,
          'splitting_feature': splitting_feature,
          'left': left_tree,
          'right': right_tree}


def count_nodes(tree):
  if tree['is_leaf']:
    return 1
  return 1 + count_nodes(tree['left']) + count_nodes(tree['right'])


def classify(tree, x, annotate=False):
  # If the node is a leaf node.
  if tree['is_leaf']:
    if annotate:
      print "At leaf, predicting %s" % tree['prediction']
    return tree['prediction']
  else:
    # Split on feature.
    split_feature_value = x[tree['splitting_feature']]
    if annotate:
      print "Split on %s = %s" % (tree['splitting_feature'], split_feature_value)
    if split_feature_value == 0:
      return classify(tree['left'], x, annotate)
    else:
      return classify(tree['right'], x, annotate)


def evaluate_classification_error(tree, data):
  # Apply the classify(tree, x) to each row in your data
  prediction = data.apply(lambda x: classify(tree, x), axis=1)
  # Once you've made the predictions, calculate the classification error
  return (data[target] != np.array(prediction)).values.sum() / float(len(data))


def adaboost_with_tree_stumps(data, features, target, num_tree_stumps):
  # start with unweighted data
  alpha = np.array([1.] * len(data))
  weights = []
  tree_stumps = []
  target_values = data[target]

  for t in xrange(num_tree_stumps):
    print '====================================================='
    print 'Adaboost Iteration %d' % t
    print '====================================================='
    # Learn a weighted decision tree stump. Use max_depth=1

    tree_stump = weighted_decision_tree_create(data, features, target, data_weights=alpha, max_depth=1)

    tree_stumps.append(tree_stump)

    # Make predictions
    predictions = data.apply(lambda x: classify(tree_stump, x), axis=1)

    # Produce a Boolean array indicating whether
    # each data point was correctly classified
    is_correct = predictions == target_values
    is_wrong = predictions != target_values

    # Compute weighted error
    weighted_error = np.sum(np.array(is_wrong) * alpha) * 1. / np.sum(alpha)

    # Compute model coefficient using weighted error
    weight = 1. / 2 * log((1 - weighted_error) * 1. / (weighted_error))
    weights.append(weight)

    # Adjust weights on data point
    adjustment = is_correct.apply(lambda is_correct: exp(-weight) if is_correct else exp(weight))

    # Scale alpha by multiplying by adjustment
    # Then normalize data points weights
    alpha = alpha * np.array(adjustment)
    alpha = alpha / np.sum(alpha)

  return weights, tree_stumps



def predict_adaboost(stump_weights, tree_stumps, data):
    scores = np.array([0.]*len(data))
    
    for i, tree_stump in enumerate(tree_stumps):
        predictions = data.apply(lambda x: classify(tree_stump, x), axis=1)
        
        # Accumulate predictions on scores array
        scores = scores + stump_weights[i] * np.array(predictions)
    
    # return the prediction 
    return np.array(1 * (scores > 0) + (-1) * (scores <= 0))


def print_stump(tree):
    split_name = tree['splitting_feature'] # split_name is something like 'term. 36 months'
    if split_name is None:
        print "(leaf, label: %s)" % tree['prediction']
        return None
    split_feature, split_value = split_name.split('_')
    print '                       root'
    print '         |---------------|----------------|'
    print '         |                                |'
    print '         |                                |'
    print '         |                                |'
    print '  [{0} == 0]{1}[{0} == 1]    '.format(split_name, ' '*(27-len(split_name)))
    print '         |                                |'
    print '         |                                |'
    print '         |                                |'
    print '    (%s)                 (%s)' \
        % (('leaf, label: ' + str(tree['left']['prediction']) if tree['left']['is_leaf'] else 'subtree'),
           ('leaf, label: ' + str(tree['right']['prediction']) if tree['right']['is_leaf'] else 'subtree'))



# Checking your Adaboost code
stump_weights, tree_stumps = adaboost_with_tree_stumps(train_data, features, target, num_tree_stumps=2)
print_stump(tree_stumps[0])
print_stump(tree_stumps[1])
print stump_weights


# Training a boosted ensemble of 10 stumps
stump_weights, tree_stumps = adaboost_with_tree_stumps(train_data, features, 
                                target, num_tree_stumps=10)
traindata_predictions = predict_adaboost(stump_weights, tree_stumps, train_data)
train_accuracy = np.sum(np.array(train_data[target]) == traindata_predictions) / float(len(traindata_predictions))
test_predictions = predict_adaboost(stump_weights, tree_stumps, test_data)
test_accuracy = np.sum(np.array(test_data[target]) == test_predictions) / float(len(test_predictions))

print(train_accuracy, test_accuracy)
print(stump_weights)
plt.plot(stump_weights)
plt.show()



#######################################################################################################################

# Performance plots
# How does accuracy change with adding stumps to the ensemble?
stump_weights, tree_stumps = adaboost_with_tree_stumps(train_data, 
                                 features, target, num_tree_stumps=30)


# Computing training error at the end of each iteration
error_all = []
for n in xrange(1, 31):
    predictions = predict_adaboost(stump_weights[:n], tree_stumps[:n], train_data)
    error = np.sum(np.array(train_data[target]) != predictions) / float(len(predictions))
    error_all.append(error)
    print "Iteration %s, training error = %s" % (n, error_all[n-1])


# Evaluation on the test data
test_error_all = []
for n in xrange(1, 31):
    predictions = predict_adaboost(stump_weights[:n], tree_stumps[:n], test_data)
    error = np.sum(np.array(test_data[target]) != predictions) / float(len(predictions))
    test_error_all.append(error)
    print "Iteration %s, test error = %s" % (n, test_error_all[n-1])


# plot errors
plt.rcParams['figure.figsize'] = 7, 5
plt.plot(range(1,31), error_all, '-', linewidth=4.0, label='Training error')
plt.plot(range(1,31), test_error_all, '-', linewidth=4.0, label='Test error')
plt.title('Performance of Adaboost ensemble')
plt.xlabel('# of iterations')
plt.ylabel('Classification error')
plt.rcParams.update({'font.size': 16})
plt.legend(loc='best', prop={'size':15})
plt.tight_layout()
plt.show()