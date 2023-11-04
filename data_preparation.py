# data_preparation.py
# Python script for data cleaning, preprocessing, and feature engineering using Pandas and NumPy

import pandas as pd
import numpy as np

# Load the raw data
raw_data = pd.read_csv('data/raw_data.csv')

# Data cleaning and preprocessing
# Example:
# Handle missing values by filling them with the mean of the column
raw_data.fillna(raw_data.mean(), inplace=True)

# Remove duplicate rows
raw_data.drop_duplicates(inplace=True)

# Normalize numeric columns (e.g., using Min-Max scaling)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
numeric_columns = ['age', 'income', 'other_numeric_feature']
raw_data[numeric_columns] = scaler.fit_transform(raw_data[numeric_columns])

# Encode categorical features using one-hot encoding
raw_data = pd.get_dummies(raw_data, columns=['categorical_feature'])

# Feature engineering
# Example:
# Create a new feature by combining existing ones
raw_data['feature_sum'] = raw_data['feature1'] + raw_data['feature2']

# Save the processed data
processed_data = raw_data

# Save the processed data to a new CSV file
processed_data.to_csv('data/processed_data.csv', index=False)
