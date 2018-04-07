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

loans = pd.read_csv('lending-club-data.csv')
loans['safe_loans'] = loans['bad_loans'].apply(lambda x: +1 if x == 0 else -1)
del loans['bad_loans']

all_loans = loans.shape[0]
safe_loans = loans[loans['safe_loans'] == 1].shape[0]
risky_loans = all_loans - safe_loans
safe_loans_percentage = 1.0 * safe_loans / all_loans * 100.0
risky_loans_percentage = 1.0 * risky_loans / all_loans * 100.0

features = ['grade',                     # grade of the loan
            'sub_grade',                 # sub-grade of the loan
            'short_emp',                 # one year or less of employment
            'emp_length_num',            # number of years of employment
            'home_ownership',            # home_ownership status: own, mortgage or rent
            'dti',                       # debt to income ratio
            'purpose',                   # the purpose of the loan
            'term',                      # the term of the loan
            'last_delinq_none',          # has borrower had a delinquincy
            'last_major_derog_none',     # has borrower had 90 day or worse rating
            'revol_util',                # percent of available credit being used
            'total_rec_late_fee',        # total late fees received to day
            ]
target = 'safe_loans'                    # prediction target (y) (+1 means safe, -1 is risky)
loans = loans[features + [target]]


# one-hot encoding
loans_categ = loans.select_dtypes(include=[object])
le = preprocessing.LabelEncoder()
loans_categ = loans_categ.apply(le.fit_transform)

for col in loans_categ.columns.values:
  col_name = col + '_'
  unique1 = loans[col].unique()
  unique2 = loans_categ[col].unique()
  for u1, u2 in zip(unique1, unique2):
    new_col_name = col_name + u1
    loans[new_col_name] = loans_categ[col].apply(lambda x: 1 if x == u2 else 0)
  del loans[col]

print(len(loans.columns.values))

##############################################################################################################

# train-validation sets
with open('module-5-assignment-1-train-idx.json', 'r') as f:
  train_json = json.load(f)
with open('module-5-assignment-1-validation-idx.json', 'r') as f:
  validation_json = json.load(f)

train_data = loans.iloc[train_json]
validation_data = loans.iloc[validation_json]


# Build a decision tree classifier
train_Y = train_data['safe_loans'].as_matrix()
train_X = train_data.drop('safe_loans', axis=1).as_matrix()
del train_data['safe_loans']

decision_tree_model = DecisionTreeClassifier(max_depth=6)
decision_tree_model = decision_tree_model.fit(train_X, train_Y)

small_model = DecisionTreeClassifier(max_depth=2)
small_model = small_model.fit(train_X, train_Y)


# Visualizing a learned model

dot_data = tree.export_graphviz(small_model, out_file='simple_tree.dot',
                                feature_names=train_data.columns,
                                class_names=['+1', '-1'],
                                filled=True, rounded=True,
                                special_characters=True)
system("dot -Tpng simple_tree.dot -o simple_tree.png")


validation_safe_loans = validation_data[validation_data[target] == 1]
validation_risky_loans = validation_data[validation_data[target] == -1]
sample_validation_data_risky = validation_risky_loans[0:2]
sample_validation_data_safe = validation_safe_loans[0:2]
sample_validation_data = sample_validation_data_safe.append(sample_validation_data_risky)

sample_validation_data_Y = sample_validation_data['safe_loans'].as_matrix()
sample_validation_data_X = sample_validation_data.drop('safe_loans', axis=1).as_matrix()
predicted = decision_tree_model.predict(sample_validation_data_X)
predicted_probability = decision_tree_model.predict_proba(sample_validation_data_X)
print(sample_validation_data_Y)
print(predicted)
print(predicted_probability)

predicted2 = small_model.predict(sample_validation_data_X)
predicted_probability2 = small_model.predict_proba(sample_validation_data_X)
print(predicted2)
print(predicted_probability2)


# Accuracy on training data
ac1 = decision_tree_model.score(train_X, train_Y)
ac2 = small_model.score(train_X, train_Y)
print(ac1, ac2)

# Accuracy on validation data
validation_Y = validation_data['safe_loans'].as_matrix()
validation_X = validation_data.drop('safe_loans', axis=1).as_matrix()
ac1 = decision_tree_model.score(validation_X, validation_Y)
ac2 = small_model.score(validation_X, validation_Y)
print(ac1, ac2)

# Evaluating accuracy of a complex decision tree model
big_model = DecisionTreeClassifier(max_depth=10)
big_model = big_model.fit(train_X, train_Y)
ac1 = big_model.score(train_X, train_Y)
ac2 = big_model.score(validation_X, validation_Y)
print(ac1, ac2)

# Quantifying the cost of mistakes
predictions = decision_tree_model.predict(validation_X)
false_positives = ((predictions == 1) * (validation_Y == -1)).sum()
false_negatives = ((predictions == -1) * (validation_Y == 1)).sum()
correct_predictions = (predictions == validation_Y).sum()
print(false_positives)
print(false_negatives)
print(correct_predictions)
