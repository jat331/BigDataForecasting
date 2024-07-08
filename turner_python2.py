#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 13:16:19 2024

@author: jamesturner
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load the dataset
nfldata = pd.read_csv("/Users/jamesturner/Downloads/PastFiveSuperBowls2.csv")

# Normalize the data
for col in nfldata.select_dtypes(include=[np.number]):
    if col != "Super Bowl":  
        min_val = nfldata[col].min()
        max_val = nfldata[col].max()
        nfldata[col] = (nfldata[col] - min_val) * 10 / (max_val - min_val)

# Separate this year's team
this_years_team = nfldata.head(2)

# Drop N/A Values
nfldata = nfldata.dropna()

# Prepare Data
X = nfldata.drop(['Win', 'Team', 'Super Bowl'], axis=1)
y = nfldata['Win']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Ridge Regression with Cross-Validation
alphas = [10, 100, 200, 500, 1000, 2000, 5000]
ridge_reg = RidgeCV(alphas=alphas)
ridge_reg.fit(X_train, y_train)

# Make predictions on the test set
y_pred_test = ridge_reg.predict(X_test)

# Calculate RMSE on the test set
ridge_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
print("Ridge Regression RMSE on Test Set:", ridge_rmse)

# Make predictions for this year's teams
X_this_year = this_years_team.drop(['Win', 'Team', 'Super Bowl'], axis=1)
X_this_year_scaled = scaler.transform(X_this_year)
y_pred_this_year = ridge_reg.predict(X_this_year_scaled)

# Apply the sigmoid function to get probabilities
probabilities_this_year = 1 / (1 + np.exp(-y_pred_this_year))

# Print out predictions for this year's teams
for team, probability in zip(this_years_team['Team'], probabilities_this_year):
    print(f"Team: {team}, Probability of Winning Super Bowl: {probability}")
  
    
