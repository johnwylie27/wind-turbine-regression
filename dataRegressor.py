# John Wylie, RIN# 661262436
# This script performs a linear regression on the data files formatted in readData.py

import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt

## Load Data
dataFileX = 'C:\\Users\\John Wylie\\Documents\\RPI\\Courses\\MANE 6962 Machine Learning\\Project Git\\wind-turbine-regression\\df_data_X.csv'
dataFiley = 'C:\\Users\\John Wylie\\Documents\\RPI\\Courses\\MANE 6962 Machine Learning\\Project Git\\wind-turbine-regression\\df_data_y.csv'
dfX1 = pd.read_csv(dataFileX, sep=',', index_col=0)
dfy = pd.read_csv(dataFiley, sep=',', index_col=0)

# Full Length Dataset
X1 = pd.DataFrame.to_numpy(dfX1, na_value=0) # make into numpy array
y = pd.DataFrame.to_numpy(dfy, na_value=0) # make into numpy array

# Abridged Dataset without Pressure Tap Data
dfX2 = pd.concat([dfX1[dfX1.columns[:5]], dfX1[dfX1.columns[-4:]]], axis=1)
X2 = pd.DataFrame.to_numpy(dfX2) # make into numpy array
# We don't need to repeat for y b/c it is the same for both full and abridged

## Preprocess Data
# for i in range():