# John Wylie, RIN# 661262436
# This script performs a linear regression on the data files formatted in readData.py

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

## Starting Parameters
pp_plt = 0 # boolean to determine whether to plot pairplot figure
nn = 1 # boolean to determine whether to run the neural network
FS = 15 # font size for plotting labels

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
# Get shapes of the data arrays
m1, n1 = X1.shape
m2, n2 = X2.shape
o, p = y.shape

# Make copies of the old non-normalized data
X1_original = X1.copy() # have to make copy of np array to specify different address in memory
X2_original = X2.copy() # otherwise all the changes to X2 will apply to X2_original!
y_original = y.copy()

# Normalize all values
for i in range(n1):
    X1[:,i] = (X1[:,i] - np.mean(X1[:,i])) / np.std(X1[:,i])
for i in range(n2):
    X2[:,i] = (X2[:,i] - np.mean(X2[:,i])) / np.std(X2[:,i])
for i in range(p):
    y[:,i] = (y[:,i] - np.mean(y[:,i])) / np.std(y[:,i])

## Visualize the Data
if pp_plt == 1:
    sns.pairplot(dfX2, kind='scatter')

## Split Data into Training and Testing Splits
X1_train, X1_test, y_train, y_test = train_test_split(X1, y, test_size=0.3, random_state=123)
X2_train, X2_test, y_train, y_test = train_test_split(X2, y, test_size=0.3, random_state=123)

if nn == 1:
    alpha = 0.01 # learning rate for the optimizer
    epo = 400 # number of epochs
    
    model = Sequential()
    model.add(Dense(10, input_dim=n2, activation='relu')) # one input neuron
    model.add(Dense(10, activation='relu')) # two neurons in single hidden layer
    model.add(Dense(3)) # one output neuron
    opt = tf.keras.optimizers.experimental.SGD(learning_rate=alpha, momentum=0) # stochastic gradient descent optimizer
    mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM) # mean squared error for loss calculation
    model.compile(loss=mse, optimizer=opt, metrics=['mse', 'mae', 'mape'])
    tr_history2 = model.fit(X2_train, y_train, epochs=epo, validation_data=(X2_test, y_test), verbose=0, use_multiprocessing=-3) # validation for monitoring validation loss and metrics at the end of each epoch
    model.summary()
    y_pred2 = model.predict(X2_test, verbose=0, use_multiprocessing=-3)
    test_loss2 = mean_squared_error(y_test, y_pred2)
    print(test_loss2)
    
    # Plot Results
    plt.figure(figsize=(8, 5))
    plt.plot(tr_history2.epoch, tr_history2.history['loss'], 'o-', linewidth=0.5, markersize=2, label=('training loss'))
    plt.plot(tr_history2.epoch, tr_history2.history['val_loss'], 'o-', linewidth=0.5, markersize=2, label=('validation loss'))
    plt.xlabel('number of epochs', fontsize = FS)
    plt.ylabel('loss', fontsize = FS)
    plt.title('Training/Testing Loss Comparison', fontsize = FS)
    plt.legend(loc='upper right')
    
    plt.figure(figsize=(8, 5))
    plt.plot(tr_history2.epoch, tr_history2.history['mae'], 'o-', linewidth=0.5, markersize=2, label=('training mse'))
    plt.plot(tr_history2.epoch, tr_history2.history['val_mae'], 'o-', linewidth=0.5, markersize=2, label=('validation mse'))
    plt.xlabel('number of epochs', fontsize = FS)
    plt.ylabel('loss metric: mae', fontsize = FS)
    plt.title('Training/Testing Loss Comparison', fontsize = FS)
    plt.legend(loc='upper right')