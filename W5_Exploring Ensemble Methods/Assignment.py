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
from sklearn.ensemble import GradientBoostingClassifier


def make_figure(dim, title, xlabel, ylabel, legend):
  plt.rcParams['figure.figsize'] = dim
  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  if legend is not None:
    plt.legend(loc=legend, prop={'size': 15})
  plt.rcParams.update({'font.size': 16})
  plt.tight_layout()
  plt.show()



# load data set
loans = pd.read_csv('lending-club-data.csv')
loans['safe_loans'] = loans['bad_loans'].apply(lambda x: +1 if x == 0 else -1)
del loans['bad_loans']

# Selecting features
target = 'safe_loans'
features = ['grade',                     # grade of the loan (categorical)
            'sub_grade_num',             # sub-grade of the loan as a number from 0 to 1
            'short_emp',                 # one year or less of employment
            'emp_length_num',            # number of years of employment
            'home_ownership',            # home_ownership status: own, mortgage or rent
            'dti',                       # debt to income ratio
            'purpose',                   # the purpose of the loan
            'payment_inc_ratio',         # ratio of the monthly payment to income
            'delinq_2yrs',               # number of delinquincies
            'delinq_2yrs_zero',          # no delinquincies in last 2 years
            'inq_last_6mths',            # number of creditor inquiries in last 6 months
            'last_delinq_none',          # has borrower had a delinquincy
            'last_major_derog_none',     # has borrower had 90 day or worse rating
            'open_acc',                  # number of open credit accounts
            'pub_rec',                   # number of derogatory public records
            'pub_rec_zero',              # no derogatory public records
            'revol_util',                # percent of available credit being used
            'total_rec_late_fee',        # total late fees received to day
            'int_rate',                  # interest rate of the loan
            'total_rec_int',             # interest received to date
            'annual_inc',                # annual income of borrower
            'funded_amnt',               # amount committed to the loan
            'funded_amnt_inv',           # amount committed by investors for the loan
            'installment',               # monthly payment owed by the borrower
            ]


# Skipping observations with missing values
print (loans.shape)
loans = loans[[target] + features].dropna()
print (loans.shape)

# one-hot encoding
categorical_variables = []
for feat_name, feat_type in zip(loans.columns, loans.dtypes):
  if feat_type == object:
    categorical_variables.append(feat_name)

for feature in categorical_variables:
  loans_one_hot_encoded = pd.get_dummies(loans[feature], prefix=feature)
  loans = loans.drop(feature, axis=1)
  for col in loans_one_hot_encoded.columns:
    loans[col] = loans_one_hot_encoded[col]

#####################################################################################################################


# train-validation sets
with open('module-8-assignment-1-train-idx.json', 'r') as f:
  train_json = json.load(f)
with open('module-8-assignment-1-validation-idx.json', 'r') as f:
  validation_json = json.load(f)
train_data = loans.iloc[train_json]
validation_data = loans.iloc[validation_json]

# Gradient boosted tree classifier
train_Y = train_data['safe_loans'].as_matrix()
train_X = train_data.drop('safe_loans', axis=1).as_matrix()
model_5 = GradientBoostingClassifier(n_estimators=5, max_depth=6).fit(train_X, train_Y)


# Making predictions
# Select all positive and negative examples.
validation_safe_loans = validation_data[validation_data[target] == 1]
validation_risky_loans = validation_data[validation_data[target] == -1]

# Select 2 examples from the validation set for positive & negative loans
sample_validation_data_risky = validation_risky_loans[0:2]
sample_validation_data_safe = validation_safe_loans[0:2]
sample_validation_data = sample_validation_data_safe.append(sample_validation_data_risky)

predicted1 = sample_validation_data['safe_loans'].as_matrix()
predicted2 = model_5.predict(sample_validation_data.drop('safe_loans', axis=1).as_matrix())
probabilities = model_5.predict_proba(sample_validation_data.drop('safe_loans', axis=1).as_matrix())
print(predicted1)
print(predicted2)
print(probabilities)


# Evaluating the model on the validation data
validation_Y = validation_data['safe_loans'].as_matrix()
validation_X = validation_data.drop('safe_loans', axis=1).as_matrix()
accuracy = model_5.score(validation_X, validation_Y) * 100
probabilities = model_5.predict_proba(validation_X)
probabilities = probabilities[:, 0]
validation_data['predictions'] = probabilities
highest_5_predictions = validation_data.sort_values('predictions').head(5)
print(accuracy)
print(highest_5_predictions['predictions'])


#####################################################################################################################


# Effect of adding more trees
model_10 = GradientBoostingClassifier(n_estimators=10, max_depth=6).fit(train_X, train_Y)
model_50 = GradientBoostingClassifier(n_estimators=50, max_depth=6).fit(train_X, train_Y)
model_100 = GradientBoostingClassifier(n_estimators=100, max_depth=6).fit(train_X, train_Y)
model_200 = GradientBoostingClassifier(n_estimators=200, max_depth=6).fit(train_X, train_Y)
model_500 = GradientBoostingClassifier(n_estimators=500, max_depth=6).fit(train_X, train_Y)


# Plot the training and validation error vs. number of trees

train_err_10 = 1 - model_10.score(train_X, train_Y)
train_err_50 = 1 - model_50.score(train_X, train_Y)
train_err_100 = 1 - model_100.score(train_X, train_Y)
train_err_200 = 1 - model_200.score(train_X, train_Y)
train_err_500 = 1 - model_500.score(train_X, train_Y)
training_errors = [train_err_10, train_err_50, train_err_100,
                   train_err_200, train_err_500]


validation_err_10 = 1 - model_10.score(validation_X, validation_Y)
validation_err_50 = 1 - model_50.score(validation_X, validation_Y)
validation_err_100 = 1 - model_100.score(validation_X, validation_Y)
validation_err_200 = 1 - model_200.score(validation_X, validation_Y)
validation_err_500 = 1 - model_500.score(validation_X, validation_Y)
validation_errors = [validation_err_10, validation_err_50, validation_err_100,
                     validation_err_200, validation_err_500]


n_trees = [10, 50, 100, 200, 500]
plt.plot([10, 50, 100, 200, 500], training_errors, linewidth=4.0, label='Training error')
plt.plot([10, 50, 100, 200, 500], validation_errors, linewidth=4.0, label='Validation error')
make_figure(dim=(10, 5), title='Error vs number of trees',
            xlabel='Number of trees',
            ylabel='Classification error',
            legend='best')
