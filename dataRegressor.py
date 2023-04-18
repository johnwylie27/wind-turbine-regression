# John Wylie, RIN# 661262436
# This script performs a linear regression on the data files formatted in readData.py

import sys
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
from keras import optimizers, regularizers

## Starting Parameters
pp_plt = 0 # boolean to determine whether to plot pairplot figure
nn1 = True # boolean to determine whether to run the NN on the full data
nn2 = True # boolean to determine whether to run the NN on the partial data
nn3 = True # boolean to determine whether to run the NN on the pressure port data only
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
X1_tr, X1_test, y_tr, y_test = train_test_split(X1, y, test_size=0.3)#, random_state=123)
X2_tr, X2_test, y_tr, y_test = train_test_split(X2, y, test_size=0.3)#, random_state=123)
X3_tr, X3_test, y_tr, y_test = train_test_split(X3, y, test_size=0.3)#, random_state=123)

## Neural Networks
actf = 'relu'
if nn1: # Run NN on data with pressure ports
    alpha = 0.005 # learning rate for the optimizer
    epo = 200 # number of epochs
    model = Sequential()
    model.add(Dense(100, input_dim=n1, activation=actf)) # one input neuron
    model.add(Dense(100, activation=actf)) # single hidden layer
    model.add(Dense(3)) # one output neuron
    opt = tf.keras.optimizers.experimental.SGD(learning_rate=alpha, momentum=0) # stochastic gradient descent optimizer
    mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM) # mean squared error for loss calculation
    model.compile(loss=mse, optimizer=opt, metrics=['mse', 'mae', 'mape'])
    tr_history1 = model.fit(X1_tr, y_tr, epochs=epo, validation_data=(X1_test, y_test), verbose=0, use_multiprocessing=-3) # validation for monitoring validation loss and metrics at the end of each epoch
    model.summary()
    y_pred1 = model.predict(X1_test, verbose=0, use_multiprocessing=-3)
    # test_loss1 = mean_squared_error(y_test, y_pred1)
    # print(test_loss1)
    
    # Plot Results
    plt.figure(figsize=(8, 5))
    plt.plot(tr_history1.epoch, tr_history1.history['loss'], 'o-', linewidth=0.5, markersize=2, label=('training loss'))
    plt.plot(tr_history1.epoch, tr_history1.history['val_loss'], 'o-', linewidth=0.5, markersize=2, label=('validation loss'))
    plt.xlabel('number of epochs', fontsize = FS)
    plt.ylabel('loss', fontsize = FS)
    plt.ylim([0, 1.2*np.max(tr_history1.history['val_loss'])])
    plt.title('Training/Testing Loss Comparison: Full Data', fontsize = FS)
    plt.legend(loc='upper right')
    # plt.savefig('G:\\My Drive\\RPI\\MANE 6962 Machine Learning\\Project\\Figures\\X1_loss.png', dpi=300)
    
    plt.figure(figsize=(8, 5))
    plt.plot(tr_history1.epoch, tr_history1.history['mae'], 'o-', linewidth=0.5, markersize=2, label=('training mse'))
    plt.plot(tr_history1.epoch, tr_history1.history['val_mae'], 'o-', linewidth=0.5, markersize=2, label=('validation mse'))
    plt.xlabel('number of epochs', fontsize = FS)
    plt.ylabel('loss metric: mae', fontsize = FS)
    plt.ylim([0, 1.2*np.max(tr_history1.history['val_mae'])])
    plt.title('Training/Testing Loss Comparison: Full Data', fontsize = FS)
    plt.legend(loc='upper right')
    # plt.savefig('G:\\My Drive\\RPI\\MANE 6962 Machine Learning\\Project\\Figures\\X1_mae.png', dpi=300)
    
    plt.figure(figsize=(8, 5))
    plt.plot(y_test[:,0], y_test[:,1], 'ko', label='Test Data')
    plt.plot(y_pred1[:,0], y_pred1[:,1], 'r*', label='Predictions')
    
if nn2: # Run NN on data without pressure ports
    alpha = 0.03 # learning rate for the optimizer
    epo = 200 # number of epochs
    nneur = 50
    model = Sequential()
    model.add(Dense(nneur, input_dim=n2, activation=actf, kernel_regularizer=regularizers.L1(1e-4))) # one input neuron
    model.add(Dense(nneur, activation=actf, kernel_regularizer=regularizers.L1(1e-4))) # repeat alpha number of times
    model.add(Dense(nneur, activation=actf, kernel_regularizer=regularizers.L1(1e-4))) # repeat alpha number of times
    model.add(Dense(nneur, activation=actf, kernel_regularizer=regularizers.L1(1e-4))) # repeat alpha number of times
    model.add(Dense(nneur, activation=actf, kernel_regularizer=regularizers.L1(1e-4))) # repeat alpha number of times
    model.add(Dense(3)) # one output neuron
    opt = tf.keras.optimizers.experimental.RMSprop(learning_rate=alpha, momentum=0) # stochastic gradient descent optimizer
    mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM) # mean squared error for loss calculation
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
    model.compile(loss=mse, optimizer=opt, metrics=['mse', 'mae', 'mape'])
    tr_history2 = model.fit(X2_tr, y_tr, epochs=epo, validation_data=(X2_test, y_test), callbacks=[callback], verbose=0, use_multiprocessing=-1) # validation for monitoring validation loss and metrics at the end of each epoch
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
    plt.ylim([0, 1.5*np.max(tr_history2.history['val_loss'])])
    plt.title('Training/Testing Loss Comparison: Partial Data', fontsize = FS)
    plt.legend(loc='upper right')
    plt.savefig('G:\\My Drive\\RPI\\MANE 6962 Machine Learning\\Project\\Figures\\X2_loss.png', dpi=300)
    
    plt.figure(figsize=(8, 5))
    plt.plot(tr_history2.epoch, tr_history2.history['mse'], 'o-', linewidth=0.5, markersize=2, label=('training mse'))
    plt.plot(tr_history2.epoch, tr_history2.history['val_mse'], 'o-', linewidth=0.5, markersize=2, label=('validation mse'))
    plt.xlabel('number of epochs', fontsize = FS)
    plt.ylabel('loss metric: mse', fontsize = FS)
    plt.ylim([0, 1.5*np.max(tr_history2.history['val_mse'])])
    plt.title('Training/Testing Loss Comparison: Partial Data', fontsize = FS)
    plt.legend(loc='upper right')
    # plt.savefig('G:\\My Drive\\RPI\\MANE 6962 Machine Learning\\Project\\Figures\\X2_mse.png', dpi=300)
    
    
    plt.figure(figsize=(8, 5))
    plt.plot(y_test[:,0], y_test[:,1], 'ko', label='Test Data')
    plt.plot(y_pred2[:,0], y_pred2[:,1], 'r*', label='Predictions')
    
if nn3: # Run NN on pressure port data only
    alpha = 0.03 # learning rate for the optimizer
    epo = 200 # number of epochs
    nneur = 50
    model = Sequential()
    model.add(Dense(nneur, input_dim=n2, activation=actf, kernel_regularizer=regularizers.L1(1e-4))) # one input neuron
    model.add(Dense(nneur, activation=actf, kernel_regularizer=regularizers.L1(1e-4))) # repeat alpha number of times
    model.add(Dense(nneur, activation=actf, kernel_regularizer=regularizers.L1(1e-4))) # repeat alpha number of times
    model.add(Dense(nneur, activation=actf, kernel_regularizer=regularizers.L1(1e-4))) # repeat alpha number of times
    model.add(Dense(nneur, activation=actf, kernel_regularizer=regularizers.L1(1e-4))) # repeat alpha number of times
    model.add(Dense(3)) # one output neuron
    opt = tf.keras.optimizers.experimental.RMSprop(learning_rate=alpha, momentum=0) # stochastic gradient descent optimizer
    mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM) # mean squared error for loss calculation
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
    model.compile(loss=mse, optimizer=opt, metrics=['mse', 'mae', 'mape'])
    tr_history3 = model.fit(X3_tr, y_tr, epochs=epo, validation_data=(X3_test, y_test), verbose=0, use_multiprocessing=-3)
    model.summary()
    y_pred3 = model.predict(X3_test, verbose=0, use_multiprocessing=-3)
    # test_loss3 = mean_squared_error(y_test, y_pred3)
    # print(test_loss3)
    
    # Plot Results
    plt.figure(figsize=(8, 5))
    plt.plot(tr_history3.epoch, tr_history3.history['loss'], 'o-', linewidth=0.5, markersize=2, label=('training loss'))
    plt.plot(tr_history3.epoch, tr_history3.history['val_loss'], 'o-', linewidth=0.5, markersize=2, label=('validation loss'))
    plt.xlabel('number of epochs', fontsize = FS)
    plt.ylabel('loss', fontsize = FS)
    plt.ylim([0, 1.2*np.max(tr_history3.history['val_loss'])])
    plt.title('Training/Testing Loss Comparison: Partial Data', fontsize = FS)
    plt.legend(loc='upper right')
    # plt.savefig('G:\\My Drive\\RPI\\MANE 6962 Machine Learning\\Project\\Figures\\X3_loss.png', dpi=300)
    
    plt.figure(figsize=(8, 5))
    plt.plot(tr_history3.epoch, tr_history3.history['mae'], 'o-', linewidth=0.5, markersize=2, label=('training mse'))
    plt.plot(tr_history3.epoch, tr_history3.history['val_mae'], 'o-', linewidth=0.5, markersize=2, label=('validation mse'))
    plt.xlabel('number of epochs', fontsize = FS)
    plt.ylabel('loss metric: mae', fontsize = FS)
    plt.ylim([0, 1.2*np.max(tr_history3.history['val_mae'])])
    plt.title('Training/Testing Loss Comparison: Partial Data', fontsize = FS)
    plt.legend(loc='upper right')
    # plt.savefig('G:\\My Drive\\RPI\\MANE 6962 Machine Learning\\Project\\Figures\\X3_mae.png', dpi=300)
    
    plt.figure(figsize=(8, 5))
    plt.plot(y_test[:,0], y_test[:,1], 'ko', label='Test Data')
    plt.plot(y_pred3[:,0], y_pred3[:,1], 'r*', label='Predictions')