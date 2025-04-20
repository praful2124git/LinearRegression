from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('data.csv')
print(data.head())

# Assuming the dataset has columns 'feature' and 'target'
X = data[['feature']]
y = data['target']
# Split the dataset into training and testing sets
print("Splitting the dataset into training and testing sets...")