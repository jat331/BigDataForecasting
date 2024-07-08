#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 18:21:46 2024

@author: jamesturner
"""
# Wine Quality Prediction 


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import csv

# Download the data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
wine_data = pd.read_csv(url, sep=';')


# Convert wine quality to binary (1 for good, 0 for not good)
wine_data['quality_binary'] = (wine_data['quality'] >= 7).astype(int)

# Display the first few rows of the modified dataset
first_few_rows_modified = wine_data.head()
print(first_few_rows_modified)

# Display basic summary statistics for the modified dataset
summary_stats_modified = wine_data.describe()
print(summary_stats_modified)

# Plotting data for the modified dataset
wine_data['quality_binary'].hist(figsize=(6, 4))
plt.title('Distribution of Wine Quality (Binary)')
plt.xlabel('Quality Binary')
plt.ylabel('Count')
plt.show()

# Define the target variable (binary)
target_variable_binary = 'quality_binary'

# Separate features and target variable
X_binary = wine_data.drop(['quality', 'quality_binary'], axis=1)
y_binary = wine_data[target_variable_binary]

# Split the data into training and testing sets for binary classification
X_train_binary, X_test_binary, y_train_binary, y_test_binary = train_test_split(
    X_binary, y_binary, test_size=0.2, random_state=42
)

# Feature scaling for binary classification (0-1 standardization)
scaler_binary = MinMaxScaler()  # Use MinMaxScaler for 0-1 scaling
X_train_scaled_binary = scaler_binary.fit_transform(X_train_binary)
X_test_scaled_binary = scaler_binary.transform(X_test_binary)

# Loop through different k values for kNN
for k in range(1, 11):
    # Build and train kNN classifier (remove random_state)
    knn_classifier_binary = KNeighborsClassifier(n_neighbors=k)  # Removed random_state
    knn_classifier_binary.fit(X_train_scaled_binary, y_train_binary)

    # Cross-validation for kNN
    cv_scores_knn = cross_val_score(knn_classifier_binary, X_train_scaled_binary, y_train_binary, cv=5)

    # Calculate and print average accuracy
    average_accuracy = cv_scores_knn.mean()
    print(f"k = {k}, Average Accuracy: {average_accuracy:.4f}")

# Naive Bayes classifier evaluation
naive_bayes_classifier_binary = GaussianNB()
naive_bayes_classifier_binary.fit(X_train_scaled_binary, y_train_binary)

# Cross-validation for Naive Bayes
cv_scores_naive_bayes = cross_val_score(naive_bayes_classifier_binary, X_train_scaled_binary, y_train_binary, cv=5)
print("\nCross-Validation Scores for Naive Bayes:")
print(cv_scores_naive_bayes)
print("Average Accuracy:", cv_scores_naive_bayes.mean())

# Predictions for binary classification
y_pred_naive_bayes_binary = naive_bayes_classifier_binary.predict(X_test_scaled_binary)
print(y_pred_naive_bayes_binary)
print(f"Accuracy: {accuracy_score(y_test_binary, y_pred_naive_bayes_binary)}")  # Added closing parenthesis
print("Confusion Matrix:\n", confusion_matrix(y_test_binary, y_pred_naive_bayes_binary))
print(classification_report(y_test_binary, y_pred_naive_bayes_binary))








