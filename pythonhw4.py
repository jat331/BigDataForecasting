#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 14:55:31 2024

@author: jamesturner
"""

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load the dataset
data = pd.read_csv('assignment4_HRemployee_attrition.csv')

# Drop specified categorical columns, which are very low on Feauture Ranking, having an importance of .00000000
drop_columns = ['PerformanceRating', 'Over18', 'StandardHours', 'EmployeeCount']
data.drop(drop_columns, axis=1, inplace=True)

# Encode categorical variables
label_encoders = {}
for column in data.select_dtypes(include=['object']):
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Split features and target variable
X = data.drop('Attrition', axis=1)
y = data['Attrition']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build Neural Network Classifier
nn_classifier = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, activation='relu', solver='adam', random_state=42)
nn_classifier.fit(X_train, y_train)

# Predict using the neural network classifier
y_pred_nn = nn_classifier.predict(X_test)

# Build XGBoost Classifier
xgb_classifier = XGBClassifier(learning_rate=0.1, n_estimators=100, max_depth=3, random_state=42)
xgb_classifier.fit(X_train, y_train)

# Get feature importances
feature_importance = xgb_classifier.feature_importances_

# Create a DataFrame to display feature importances
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Print feature importance ranking
print("Feature Importance Ranking:")
print(feature_importance_df)


# Predict using the XGBoost classifier
y_pred_xgb = xgb_classifier.predict(X_test)

data

# Evaluate the models
print("Neural Network Classifier Results:")
print(classification_report(y_test, y_pred_nn))
print("Accuracy:", accuracy_score(y_test, y_pred_nn))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_nn))

# Evaluate the models
print("\nXGBoost Classifier Results:")
print(classification_report(y_test, y_pred_xgb))
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_xgb))

