# John Wylie, RIN# 661262436
# This script performs a linear regression on the data files formatted in readData.py

import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt

dataFileX = 'C:\\Users\\John Wylie\\Documents\\RPI\\Courses\\MANE 6962 Machine Learning\\Project Git\\wind-turbine-regression\\df_data_X.csv'
dataFiley = 'C:\\Users\\John Wylie\\Documents\\RPI\\Courses\\MANE 6962 Machine Learning\\Project Git\\wind-turbine-regression\\df_data_y.csv'
dfX = pd.read_csv(dataFileX)
dfy = pd.read_csv(dataFiley)

X = pd.DataFrame.to_numpy(dfX)
y = pd.DataFrame.to_numpy(dfy)