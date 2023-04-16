# John Wylie, RIN# 661262436
# This script tunes hyperparameters for the linear regression in dataRegressor.py

import sys
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

## Starting Parameters
pp_plt = 0 # boolean to determine whether to plot pairplot figure
nn1 = 0 # boolean to determine whether to run the NN on the full data
lr = 1 # boolean to determine whether to run the Linear Regression on the full data
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

# Abridged Dataset with Pressure Ports Only
dfX3 = dfX1[dfX1.columns[5:-4]]
X3 = pd.DataFrame.to_numpy(dfX3) # make into numpy array

## Preprocess Data
# Get shapes of the data arrays
m1, n1 = X1.shape
m2, n2 = X2.shape
m3, n3 = X3.shape
o, p = y.shape

# Make copies of the old non-normalized data
X1_original = X1.copy() # have to make copy of np array to specify different address in memory
X2_original = X2.copy() # otherwise all the changes to X2 will apply to X2_original!
X3_original = X3.copy()
y_original = y.copy()

# Normalize all values: x_norm = (x-min)/(max-min)
# Puts each feature value in the domain of [0,1]
scaleX1 = np.zeros((n1, 2))
scaleX2 = np.zeros((n2, 2))
scaleX3 = np.zeros((n3, 2))
scaley = np.zeros((p, 2))
for i in range(n1):
    scaleX1[i,0] = np.min(X1[:,i])
    scaleX1[i,1] = np.max(X1[:,i])
    X1[:,i] = (X1[:,i] - scaleX1[i,0]) / (scaleX1[i,1] - scaleX1[i,0])
for i in range(n2):
    scaleX2[i,0] = np.min(X2[:,i])
    scaleX2[i,1] = np.max(X2[:,i])
    X2[:,i] = (X2[:,i] - scaleX2[i,0]) / (scaleX2[i,1] - scaleX2[i,0])
for i in range(n3):
    scaleX3[i,0] = np.min(X3[:,i])
    scaleX3[i,1] = np.max(X3[:,i])
    X3[:,i] = (X3[:,i] - scaleX3[i,0]) / (scaleX3[i,1] - scaleX3[i,0])
for i in range(p):
    scaley[i,0] = np.min(y[:,i])
    scaley[i,1] = np.max(y[:,i])
    y[:,i] = (y[:,i] - scaley[i,0]) / (scaley[i,1] - scaley[i,0])

# Shift values
c = 0
X1 = X1 + c
X2 = X2 + c
X3 = X3 + c
y = y + c

if np.isnan(np.sum(X1)):
    sys.exit('X1 has NaN value')
if np.isnan(np.sum(X2)):
    sys.exit('X2 has NaN value')
if np.isnan(np.sum(X3)):
    sys.exit('X3 has NaN value')

# Check that all data falls in the range of [0,1]
plt.figure()
for i in range(n1):
    plt.plot(X1[:,i],'--')
    plt.title('Scaled X1 Data')
    plt.xlabel('Data Index')
    plt.ylabel('Feature Value')
plt.figure()
for i in range(n2):
    plt.plot(X2[:,i],'--')
    plt.title('Scaled X2 Data')
    plt.xlabel('Data Index')
    plt.ylabel('Feature Value')
plt.figure()
for i in range(n3):
    plt.plot(X3[:,i],'--')
    plt.title('Scaled X3 Data')
    plt.xlabel('Data Index')
    plt.ylabel('Feature Value')
plt.figure()
for i in range(p):
    plt.plot(y[:,i],'--')
    plt.title('Scaled y Data')
    plt.xlabel('Data Index')
    plt.ylabel('Feature Value')

## Visualize the Data
if pp_plt == 1:
    sns.pairplot(dfX2, kind='scatter')

## Split Data into Training and Testing Splits
X1_tr, X1_test, y_tr, y_test = train_test_split(X1, y, test_size=0.3, random_state=123)
X2_tr, X2_test, y_tr, y_test = train_test_split(X2, y, test_size=0.3, random_state=123)
X3_tr, X3_test, y_tr, y_test = train_test_split(X3, y, test_size=0.3, random_state=123)

## Neural Networks
alpha = 0.01 # learning rate for the optimizer
epo = 500 # number of epochs

actf = 'relu'
if nn1 == 1: # Run NN on data with pressure ports
    model = Sequential()
    model.add(Dense(100, input_dim=n1, activation=actf)) # one input neuron
    model.add(Dense(100, activation=actf)) # two neurons in single hidden layer
    model.add(Dense(3)) # one output neuron
    opt = tf.keras.optimizers.experimental.SGD(learning_rate=alpha, momentum=0) # stochastic gradient descent optimizer
    mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM) # mean squared error for loss calculation
    model.compile(loss=mse, optimizer=opt, metrics=['mse', 'mae', 'mape'])
    tr_history1 = model.fit(X1_tr, y_tr, epochs=epo, validation_data=(X1_test, y_test), verbose=0, use_multiprocessing=-3) # validation for monitoring validation loss and metrics at the end of each epoch
    model.summary()
    y_pred1 = model.predict(X1_test, verbose=0, use_multiprocessing=-3)
    test_loss1 = mean_squared_error(y_test, y_pred1)
    print(test_loss1)
    
    # Plot Results
    plt.figure(figsize=(8, 5))
    plt.plot(tr_history1.epoch, tr_history1.history['loss'], 'o-', linewidth=0.5, markersize=2, label=('training loss'))
    plt.plot(tr_history1.epoch, tr_history1.history['val_loss'], 'o-', linewidth=0.5, markersize=2, label=('validation loss'))
    plt.xlabel('number of epochs', fontsize = FS)
    plt.ylabel('loss', fontsize = FS)
    plt.ylim([0, 1.2*np.max(tr_history1.history['val_loss'])])
    plt.title('Training/Testing Loss Comparison: Full Data', fontsize = FS)
    plt.legend(loc='upper right')
    plt.savefig('G:\\My Drive\\RPI\\MANE 6962 Machine Learning\\Project\\Figures\\X1_loss.png', dpi=300)
    
    plt.figure(figsize=(8, 5))
    plt.plot(tr_history1.epoch, tr_history1.history['mae'], 'o-', linewidth=0.5, markersize=2, label=('training mse'))
    plt.plot(tr_history1.epoch, tr_history1.history['val_mae'], 'o-', linewidth=0.5, markersize=2, label=('validation mse'))
    plt.xlabel('number of epochs', fontsize = FS)
    plt.ylabel('loss metric: mae', fontsize = FS)
    plt.ylim([0, 1.2*np.max(tr_history1.history['val_mae'])])
    plt.title('Training/Testing Loss Comparison: Full Data', fontsize = FS)
    plt.legend(loc='upper right')
        
if lr == 1: # Run linear regressor from sklearn
    lr1 = LinearRegression()
    lr1.fit(X1_tr, y_tr)
    y_lr1 = lr1.predict(X1_test)
    print(f'Coefficient of determination (R^2) score: {lr1.score(X1_test, y_test)}')
    print(f'Mean squared error: {mean_squared_error(y_test, y_lr1)}')
    print(lr1.n_features_in_)
    
    plt.figure(figsize=(8,5))
    plt.title('Lift Coefficient Weights')
    plt.plot(lr1.coef_[0,:], 'k--', label='w0')
    plt.xlabel('Feature Index')
    plt.ylabel('Coefficient Value')
    plt.figure(figsize=(8,5))
    plt.title('Drag Coefficient Weights')
    plt.plot(lr1.coef_[1,:], 'b--', label='w1')
    plt.xlabel('Feature Index')
    plt.ylabel('Coefficient Value')
    plt.figure(figsize=(8,5))
    plt.title('Moment Coefficient Weights')
    plt.plot(lr1.coef_[2,:], 'r--', label='w2')
    plt.xlabel('Feature Index')
    plt.ylabel('Coefficient Value')
    
    feat11 = np.vstack([np.arange(0, len(lr1.coef_[0,:]), 1), lr1.coef_[0,:]]).T
    feat12 = np.vstack([np.arange(0, len(lr1.coef_[0,:]), 1), lr1.coef_[1,:]]).T
    feat13 = np.vstack([np.arange(0, len(lr1.coef_[0,:]), 1), lr1.coef_[2,:]]).T
    feat11 = feat11[np.abs(feat11[:, 1]).argsort()]
    feat12 = feat12[np.abs(feat12[:, 1]).argsort()]
    feat13 = feat13[np.abs(feat13[:, 1]).argsort()]
    
    lr2 = LinearRegression()
    lr2.fit(X2_tr, y_tr)
    y_lr2 = lr2.predict(X2_test)
    print(f'Coefficient of determination (R^2) score: {lr2.score(X2_test, y_test)}')
    print(f'Mean squared error: {mean_squared_error(y_test, y_lr2)}')
    print(lr2.n_features_in_)
    
    plt.figure(figsize=(8,5))
    plt.title('Lift Coefficient Weights')
    plt.plot(lr2.coef_[0,:], 'k--', label='w0')
    plt.xlabel('Feature Index')
    plt.ylabel('Coefficient Value')
    plt.figure(figsize=(8,5))
    plt.title('Drag Coefficient Weights')
    plt.plot(lr2.coef_[1,:], 'b--', label='w1')
    plt.xlabel('Feature Index')
    plt.ylabel('Coefficient Value')
    plt.figure(figsize=(8,5))
    plt.title('Moment Coefficient Weights')
    plt.plot(lr2.coef_[2,:], 'r--', label='w2')
    plt.xlabel('Feature Index')
    plt.ylabel('Coefficient Value')
    
    feat21 = np.vstack([np.arange(0, len(lr2.coef_[0,:]), 1), lr2.coef_[0,:]]).T
    feat22 = np.vstack([np.arange(0, len(lr2.coef_[0,:]), 1), lr2.coef_[1,:]]).T
    feat23 = np.vstack([np.arange(0, len(lr2.coef_[0,:]), 1), lr2.coef_[2,:]]).T
    feat21 = feat21[np.abs(feat21[:, 1]).argsort()]
    feat22 = feat22[np.abs(feat22[:, 1]).argsort()]
    feat23 = feat23[np.abs(feat23[:, 1]).argsort()]