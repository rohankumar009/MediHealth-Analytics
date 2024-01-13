# data_preparation.py
# Python script for data cleaning, preprocessing, and feature engineering using Pandas and NumPy

import pandas as pd
import numpy as np

data = pd.read_csv('data/data.csv')

# Data cleaning and preprocessing
# Example:
# Handle missing values by filling them with the mean of the column
data.fillna(data.mean(), inplace=True)

# Remove duplicate rows
data.drop_duplicates(inplace=True)

# Normalize numeric columns (using Min-Max scaling)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
numeric_columns = ['age', 'income', 'other_numeric_feature']
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

# Encode categorical features using one-hot encoding
data = pd.get_dummies(data, columns=['categorical_feature'])

# Feature engineering
data['feature_sum'] = data['feature1'] + data['feature2']

# Save the processed data
processed_data = data

# Save the processed data to a new CSV file
processed_data.to_csv('data/processed_data.csv', index=False)
