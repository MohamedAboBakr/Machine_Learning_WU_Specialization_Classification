import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_scoreAccuracy
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def remove_punctuation(text):
  import string
  # if type(text) == float:
  # print text
  return text.translate(None, string.punctuation)


# Load amazon review dataset
products = pd.read_csv('amazon_baby.csv')

# Extract word counts and sentiments
products = products.fillna({'review': ''})
products['review_clean'] = products['review'].apply(remove_punctuation)
products = products[products['rating'] != 3]
products['sentiment'] = products['rating'].apply(lambda rating: +1 if rating > 3 else -1)


# Split data into training and test sets
with open('module-9-assignment-train-idx.json') as train_data_file:
  train_data_idx = json.load(train_data_file)
with open('module-9-assignment-test-idx.json') as test_data_file:
  test_data_idx = json.load(test_data_file)
train_data = products.iloc[train_data_idx]
test_data = products.iloc[test_data_idx]

######################################################################################################################
# Build the word count vector for each review
vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
train_matrix = vectorizer.fit_transform(train_data['review_clean'])
test_matrix = vectorizer.transform(test_data['review_clean'])


# Train a logistic regression classifier
model = LogisticRegression()
model.fit(train_matrix, train_data['sentiment'])

# Model Evaluation
accuracy = accuracy_score(y_true=test_data['sentiment'].as_matrix(), y_pred=model.predict(test_matrix))
print ("Test Accuracy: %s" % accuracy)
# Baseline: Majority class prediction
baseline = len(test_data[test_data['sentiment'] == 1]) / float(len(test_data))
print ("Baseline accuracy (majority class classifier): %s" % baseline)


# Confusion Matrix
cmat = confusion_matrix(y_true=test_data['sentiment'].as_matrix(),
                        y_pred=model.predict(test_matrix),
                        labels=model.classes_)    # use the same order of class as the LR model.
print (' target_label | predicted_label | count ')
print ('--------------+-----------------+-------')
# Print out the confusion matrix.
# NOTE: Your tool may arrange entries in a different order. Consult appropriate manuals.
for i, target_label in enumerate(model.classes_):
  for j, predicted_label in enumerate(model.classes_):
    print '{0:^13} | {1:^15} | {2:5d}'.format(target_label, predicted_label, cmat[i, j])


# Precision and Recall on testset
precision = precision_score(y_true=test_data['sentiment'].as_matrix(),
                            y_pred=model.predict(test_matrix))
print ("Precision on test data: %s" % precision)
recall = recall_score(y_true=test_data['sentiment'].as_matrix(),
                      y_pred=model.predict(test_matrix))
print ("Recall on test data: %s" % recall)


# Precision-recall tradeoff
def apply_threshold(probabilities, threshold):
  # +1 if >= threshold and -1 otherwise.
  result = np.ones(len(probabilities))
  result[probabilities < threshold] = -1
  return result


probabilities = model.predict_proba(test_matrix)[:, 1]
predictions_with_default_threshold = apply_threshold(probabilities, 0.5)
predictions_with_high_threshold = apply_threshold(probabilities, 0.9)
print (sum(probabilities >= 0.5))
print (sum(probabilities >= 0.9))


# Exploring the associated precision and recall as the threshold varies
# Threshold = 0.5
precision_with_default_threshold = precision_score(y_true=test_data['sentiment'].as_matrix(), y_pred=predictions_with_default_threshold)
recall_with_default_threshold = recall_score(y_true=test_data['sentiment'].as_matrix(), y_pred=predictions_with_default_threshold)

# Threshold = 0.9
precision_with_high_threshold = precision_score(y_true=test_data['sentiment'].as_matrix(), y_pred=predictions_with_high_threshold)
recall_with_high_threshold = recall_score(y_true=test_data['sentiment'].as_matrix(), y_pred=predictions_with_high_threshold)

print ("Precision (threshold = 0.5): %s" % precision_with_default_threshold)
print ("Recall (threshold = 0.5)   : %s" % recall_with_default_threshold)
print ("Precision (threshold = 0.9): %s" % precision_with_high_threshold)
print ("Recall (threshold = 0.9)   : %s" % recall_with_high_threshold)


#####################################################################################################################

# Precision-recall curve
threshold_values = np.linspace(0.5, 1, num=100)
precision_all = []
recall_all = []
probabilities = model.predict_proba(test_matrix)[:, 1]
for threshold in threshold_values:
  predictions = apply_threshold(probabilities, threshold)
  precision = precision_score(y_true=test_data['sentiment'].as_matrix(), y_pred=predictions)
  recall = recall_score(y_true=test_data['sentiment'].as_matrix(), y_pred=predictions)
  precision_all.append(precision)
  recall_all.append(recall)

print(precision_all[0], recall_all[0])


def plot_pr_curve(precision, recall, title):
  plt.rcParams['figure.figsize'] = 7, 5
  plt.locator_params(axis='x', nbins=5)
  plt.plot(precision, recall, 'b-', linewidth=4.0, color='#B0017F')
  plt.title(title)
  plt.xlabel('Precision')
  plt.ylabel('Recall')
  plt.rcParams.update({'font.size': 16})
  plt.show()


plot_pr_curve(precision_all, recall_all, 'Precision recall curve (all)')
print(np.array(threshold_values)[np.array(precision_all) >= 0.965])

predictions_with_098_threshold = apply_threshold(probabilities, 0.98)
sth = (np.array(test_data['sentiment'].as_matrix()) > 0) * (predictions_with_098_threshold < 0)
print (sum(sth))


cmat_098 = confusion_matrix(y_true=test_data['sentiment'].as_matrix(),
                            y_pred=predictions_with_098_threshold,
                            labels=model.classes_)    # use the same order of class as the LR model.
print(' target_label | predicted_label | count ')
print('--------------+-----------------+-------')

# Print out the confusion matrix.
# NOTE: Your tool may arrange entries in a different order. Consult appropriate manuals.
for i, target_label in enumerate(model.classes_):
  for j, predicted_label in enumerate(model.classes_):
    print ('{0:^13} | {1:^15} | {2:5d}'.format(target_label, predicted_label, cmat_098[i, j]))



#####################################################################################################################

# Precision-Recall on all baby related items
baby_reviews = test_data[test_data['name'].apply(lambda x: 'baby' in str(x).lower())]
baby_matrix = vectorizer.transform(baby_reviews['review_clean'])
probabilities = model.predict_proba(baby_matrix)[:, 1]


precision_all = []
recall_all = []

for threshold in threshold_values:
  # Make predictions. Use the `apply_threshold` function
  predictions = apply_threshold(probabilities, threshold)
  # Calculate the precision.
  precision = precision_score(y_true=baby_reviews['sentiment'].as_matrix(), y_pred=predictions)
  recall = recall_score(y_true=baby_reviews['sentiment'].as_matrix(), y_pred=predictions)
  # Append the precision and recall scores.
  precision_all.append(precision)
  recall_all.append(recall)

plot_pr_curve(precision_all, recall_all, "Precision-Recall (Baby)")
print (np.array(threshold_values)[np.array(precision_all) >= 0.965])
