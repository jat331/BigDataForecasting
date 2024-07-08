#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 11:20:55 2024

@author: jamesturner
"""

# Titanic Survival

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import statsmodels.api as sm
import matplotlib.pyplot as plt  # Added import statement for Matplotlib
import graphviz
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Load the CSV file using pandas
passenger_data = pd.read_csv("/Users/jamesturner/Downloads/assignment3.csv")
passenger_data

# Drop the specified columns
passenger_data.drop(['Ticket', 'Fare', 'Cabin','Name','Embarked'], axis=1, inplace=True)

passenger_data['Family'] = passenger_data['Parch'] + passenger_data['SibSp']

# Determine if the passenger is alone or not
passenger_data['Is_Alone'] = passenger_data['Family'] == 0

# Display the modified DataFrame
print("\nDataFrame after dropping columns:")
print(passenger_data)

# Encode categorical variables
passenger_data['Sex'] = passenger_data['Sex'].map({'male': 0, 'female': 1})


# Drop rows with missing values in the 'Age' column
passenger_data.dropna(subset=['Age'], inplace=True)

# Ensure all data types are numerical
passenger_data = passenger_data.astype(float) 

# Split data into features and target variable
X = passenger_data.drop('Survived', axis=1)
y = passenger_data['Survived']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Baseline CART model
cart_model = DecisionTreeClassifier(random_state=42)
cart_model.fit(X_train, y_train)
y_pred_cart = cart_model.predict(X_test)

# Baseline CART model
cart_model = DecisionTreeClassifier(random_state=42)
cart_scores = cross_val_score(cart_model, X_train, y_train, cv=5, scoring='accuracy')

print("Cross-Validation Scores - Baseline CART Model:")
print(cart_scores)
print("Mean Accuracy: {:.2f}".format(cart_scores.mean()))

from sklearn.model_selection import KFold

# Define number of folds for cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Store accuracy scores for each fold
probit_scores = []

# Perform cross-validation
for train_index, test_index in kfold.split(X_train):
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

    # Fit Probit model
    probit_model_fold = sm.Probit(y_train_fold, X_train_fold).fit()

    # Predict and evaluate
    y_pred_fold = probit_model_fold.predict(X_test_fold)
    y_pred_fold = [1 if x > 0.5 else 0 for x in y_pred_fold]

    accuracy_fold = accuracy_score(y_test_fold, y_pred_fold)
    probit_scores.append(accuracy_fold)

# Print cross-validation results
print("Cross-Validation Scores - Probit Model:")
print(probit_scores)
print("Mean Accuracy: {:.2f}".format(sum(probit_scores) / len(probit_scores)))

# Prune Model
tree = DecisionTreeClassifier(random_state=0)

# Fit it to the training data
tree.fit(X_train, y_train)

# Compute accuracy on the test data
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))

# Apply cost complexity pruning

# Call the cost complexity command
path = tree.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

# For each alpha, estimate the tree
clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)

# Drop the last model because that only has 1 node
clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]

# Plot accuracy (in test and training) over alpha; first compute accuracy for each alpha
train_acc = []
test_acc = []
for c in clfs:
    y_train_pred = c.predict(X_train)
    y_test_pred = c.predict(X_test)
    train_acc.append(accuracy_score(y_train_pred,y_train))
    test_acc.append(accuracy_score(y_test_pred,y_test))

# Second, plot it
plt.scatter(ccp_alphas,train_acc)
plt.scatter(ccp_alphas,test_acc)
plt.plot(ccp_alphas,train_acc,label='train_accuracy',drawstyle="steps-post")
plt.plot(ccp_alphas,test_acc,label='test_accuracy',drawstyle="steps-post")
plt.legend()
plt.title('Accuracy vs alpha')
plt.show()

# Estimate the tree with the optimal alpha and display accuracy
clf_ = DecisionTreeClassifier(random_state=0,ccp_alpha=0.01)
clf_.fit(X_train,y_train)

print("Accuracy on test set: {:.3f}".format(clf_.score(X_test, y_test)))

# Plot the pruned tree
export_graphviz(clf_, out_file="tree.dot", class_names=["malignant", "benign"],
    feature_names=X.columns, impurity=True, filled=True)

with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)

# Estimate the tree with the optimal alpha and display accuracy
clf_ = DecisionTreeClassifier(random_state=0, ccp_alpha=0.01)
clf_.fit(X_train, y_train)

# Predict labels for the test set using the pruned model
y_pred_pruned = clf_.predict(X_test)

# Print the classification report for the pruned model
print("Classification Report - Pruned Tree Model:")
print(classification_report(y_test, y_pred_pruned))

# Random Forest model
clf = RandomForestClassifier(criterion='entropy', 
                             n_estimators=700,
                             min_samples_split=10,
                             min_samples_leaf=1,
                             max_features=None,  # Change 'auto' to None or remove this parameter
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)
# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Fit the Random Forest model
clf.fit(x_train, np.ravel(y_train))

# Print accuracy of the Random Forest model
print("RF Accuracy: " + repr(round(clf.score(x_test, y_test) * 100, 2)) + "%")

# Compute cross-validated score for Random Forest model
result_rf = cross_val_score(clf, x_train, y_train, cv=10, scoring='accuracy')
print('The cross validated score for Random forest is:', round(result_rf.mean()*100, 2))

# Make predictions using cross-validated model
y_pred = cross_val_predict(clf, x_train, y_train, cv=10)

# Visualize confusion matrix for Random Forest model
sns.heatmap(confusion_matrix(y_train, y_pred), annot=True, fmt='3.0f', cmap="summer")
plt.title('Confusion_matrix for RF', y=1.05, size=15)
plt.show()
