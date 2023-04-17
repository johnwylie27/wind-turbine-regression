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
from keras.layers import Dense, Dropout
from keras import optimizers

## Starting Parameters
pp_plt = 0 # boolean to determine whether to plot pairplot figure
nn1 = 0 # boolean to determine whether to run the 1st NN hyperparameter tuning test
nn2 = 0 # boolean to determine whether to run the 2nd NN hyperparameter tuning test
nn3 = 1 # boolean to determine whether to run the 2rd NN hpyerparameter tuning test
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

actf = 'relu'
if nn1 == 1: # Run NN on data with pressure ports
    tr_history = []
    his = []
    nneur = np.array([25, 50, 100, 200]) # number of neurons per laayer (barring output layer)
    nlayers = np.array([1, 2, 3, 4]) # number of hidden layers
    # nneur = np.array([6, 7, 8]) # number of neurons per laayer (barring output layer)
    # nlayers = np.array([1, 2]) # number of hidden layers
    epo = 300 # number of epochs
    alpha = 0.01 # learning rate for optimizer
    opt = tf.keras.optimizers.legacy.SGD(learning_rate=alpha, momentum=0) # stochastic gradient descent optimizer
    mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM) # mean squared error for loss calculation
    
    tr_loss = np.zeros([len(nneur), len(nlayers)])
    v_loss = np.zeros([len(nneur), len(nlayers)])
    for i in range(len(nneur)):
        for j in range(len(nlayers)):
            model = Sequential()
            model.add(Dense(nneur[i], input_dim=n2, activation=actf)) # one input neuron
            for k in range(nlayers[j]):
                model.add(Dense(nneur[i], activation=actf)) # repeat nlayers number of times
            model.add(Dense(3)) # one output neuron
            model.compile(loss=mse, optimizer=opt, metrics=['mse', 'mae', 'mape'])
            # model.summary()
            history1 = model.fit(X2_tr, y_tr, epochs=epo, validation_data=(X2_test, y_test), verbose=0, use_multiprocessing=-3) # validation for monitoring validation loss and metrics at the end of each epoch
            history1.history['nneur'] = nneur[i]
            history1.history['nlayers'] = nlayers[j]
            his.append(history1)
            tr_loss[i,j] = np.min(history1.history['loss']) # history1.history['loss'][-1]
            v_loss[i,j] = np.min(history1.history['val_loss']) #history1.history['val_loss'][-1]
            print(f'Completed: nneur = {nneur[i]}, nlayers = {nlayers[j]} ---> val loss = {v_loss[i,j]}')

    # Training and Validation Heatmaps
    plt.figure(figsize=(12,5))
    plt.subplot(121)
    ax = sns.heatmap(tr_loss, annot=True, annot_kws={"fontsize":FS}, cbar_kws={'label': 'Training Loss'}, vmin=np.min(tr_loss), vmax=np.max(tr_loss))
    ax.set_xticklabels(nlayers, fontsize=FS*3/4);
    ax.set_yticklabels(nneur, fontsize=FS*3/4);
    plt.title('Training Losses', fontsize=FS)
    plt.xlabel('Number of Layers', fontsize=FS)
    plt.ylabel('Number of Neurons per Layer', fontsize=FS)
    plt.subplot(122)
    ax = sns.heatmap(v_loss, annot=True, annot_kws={"fontsize":FS}, cbar_kws={'label': 'Validation Loss'}, vmin=np.min(v_loss), vmax=np.max(v_loss))
    ax.set_xticklabels(nlayers, fontsize=FS*3/4);
    ax.set_yticklabels(nneur, fontsize=FS*3/4);
    plt.title('Validation Losses', fontsize=FS)
    plt.xlabel('Number of Layers', fontsize=FS)
    plt.ylabel('Number of Neurons per Layer', fontsize=FS)
    plt.savefig('G:\\My Drive\\RPI\\MANE 6962 Machine Learning\\Project\\Figures\\NNArch1.png', dpi=300)
    
    # Training and Validation Epoch Plots
    ct = 0
    co = list(plt.rcParams['axes.prop_cycle'].by_key()['color']) + ['crimson', 'indigo', 'orange', 'red', 'blue', 'green', 'brown']
    plt.figure(figsize=(12, 7))
    for i in range(len(nneur)):
        for j in range(len(nlayers)):
            plt.plot(history1.epoch, his[ct].history['loss'], '^--', linewidth=2, markersize=8, label=f'Training {nneur[i]} Neurons, {nlayers[j]} Layers', color=co[ct])
            plt.plot(history1.epoch, his[ct].history['val_loss'], 'v--', linewidth=2, markersize=8, label=f'Validation {nneur[i]} Neurons, {nlayers[j]} Layers', color=co[ct])
            ct = ct+1
    plt.xlabel('Epoch', fontsize = FS)
    plt.ylabel('Loss', fontsize = FS)
    # plt.ylim([0, 1.2*np.max(tr_history1.history['val_loss'])])
    plt.title('Training/Testing Loss Comparison', fontsize = FS)
    plt.legend(loc='upper right', ncol=2)
    plt.savefig('G:\\My Drive\\RPI\\MANE 6962 Machine Learning\\Project\\Figures\\NNArch2.png', dpi=300)
    
if nn2 == 1: # Epoch/Learning Rate
    tr_history = []
    his = []
    epoch = np.array([150, 200, 250, 300]) # number of epochs
    alpha = np.array([0.01, 0.02, 0.03]) # learning rate
    # epoch = np.array([6, 7, 8])
    # alpha = np.array([0.01, 0.02])
    nneur = 200 # number of neurons per input/hidden layer
    # nneur = 20
    mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM) # mean squared error for loss calculation
    
    tr_loss = np.zeros([len(epoch), len(alpha)])
    v_loss = np.zeros([len(epoch), len(alpha)])
    for i in range(len(epoch)):
        for j in range(len(alpha)):
            model = Sequential()
            model.add(Dense(nneur, input_dim=n2, activation=actf)) # one input neuron
            model.add(Dense(nneur, activation=actf)) # repeat alpha number of times
            model.add(Dense(nneur, activation=actf)) # repeat alpha number of times
            model.add(Dense(nneur, activation=actf)) # repeat alpha number of times
            model.add(Dense(nneur, activation=actf)) # repeat alpha number of times
            model.add(Dense(3)) # one output neuron
            opt = tf.keras.optimizers.legacy.SGD(learning_rate=alpha[j], momentum=0) # stochastic gradient descent optimizer
            model.compile(loss=mse, optimizer=opt, metrics=['mse', 'mae', 'mape'])
            # model.summary()
            history1 = model.fit(X2_tr, y_tr, epochs=epoch[i], validation_data=(X2_test, y_test), verbose=0, use_multiprocessing=-2) # validation for monitoring validation loss and metrics at the end of each epoch
            history1.history['epoch'] = epoch[i]
            history1.history['alpha'] = alpha[j]
            his.append(history1)
            tr_loss[i,j] = np.min(history1.history['loss']) # history1.history['loss'][-1]
            v_loss[i,j] = np.min(history1.history['val_loss']) #history1.history['val_loss'][-1]
            print(f'Completed: epoch = {epoch[i]}, alpha = {alpha[j]} ---> val loss = {v_loss[i,j]}')

    # Training and Validation Heatmaps
    plt.figure(figsize=(12,5))
    plt.subplot(121)
    ax = sns.heatmap(tr_loss, annot=True, annot_kws={"fontsize":FS}, cbar_kws={'label': 'Training Loss'}, vmin=np.min(tr_loss), vmax=np.max(tr_loss))
    ax.set_xticklabels(alpha, fontsize=FS*3/4);
    ax.set_yticklabels(epoch, fontsize=FS*3/4);
    plt.title('Training Losses', fontsize=FS)
    plt.xlabel(r'$Learning Rate (\alpha$)', fontsize=FS)
    plt.ylabel('Number of Epochs', fontsize=FS)
    plt.subplot(122)
    ax = sns.heatmap(v_loss, annot=True, annot_kws={"fontsize":FS}, cbar_kws={'label': 'Validation Loss'}, vmin=np.min(v_loss), vmax=np.max(v_loss))
    ax.set_xticklabels(alpha, fontsize=FS*3/4);
    ax.set_yticklabels(epoch, fontsize=FS*3/4);
    plt.title('Validation Losses', fontsize=FS)
    plt.xlabel(r'$Learning Rate (\alpha$)', fontsize=FS)
    plt.ylabel('Number of Epochs', fontsize=FS)
    plt.savefig('G:\\My Drive\\RPI\\MANE 6962 Machine Learning\\Project\\Figures\\NNEpochLR1v2.png', dpi=300)
    
    # Training and Validation Epoch Plots
    ct = 0
    co = list(plt.rcParams['axes.prop_cycle'].by_key()['color']) + ['crimson', 'indigo', 'orange', 'red', 'blue', 'green', 'brown']
    plt.figure(figsize=(12, 7))
    for i in range(len(epoch)):
        for j in range(len(alpha)):
            plt.plot(np.arange(1, his[ct].history['epoch']+1, 1), his[ct].history['loss'], '^--', linewidth=2, markersize=8, label=f'Training {epoch[i]} Epochs, {alpha[j]} Alpha', color=co[ct])
            plt.plot(np.arange(1, his[ct].history['epoch']+1, 1), his[ct].history['val_loss'], 'v--', linewidth=2, markersize=8, label=f'Validation {epoch[i]} Epochs, {alpha[j]} Alpha', color=co[ct])
            ct = ct+1
    plt.xlabel('Epoch', fontsize = FS)
    plt.ylabel('Loss', fontsize = FS)
    # plt.ylim([0, 1.2*np.max(tr_history1.history['val_loss'])])
    plt.title('Training/Testing Loss Comparison', fontsize = FS)
    plt.legend(loc='upper right', ncol=2)
    plt.savefig('G:\\My Drive\\RPI\\MANE 6962 Machine Learning\\Project\\Figures\\NNEpochLR2v2.png', dpi=300)

if nn3 == 1: # Dropout Layer 
    tr_history = []
    his = []
    ind = np.array([0, 1, 2, 3, 4]) # number of epochs
    alpha = 0.03 # learning rate
    epoch = 300 # number of epochs
    epoch = 3
    # nneur = 200 # number of neurons per input/hidden layer
    nneur = 20
    mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM) # mean squared error for loss calculation
    
    tr_loss = np.zeros([len(ind),1])
    v_loss = np.zeros([len(ind),1])
    for i in range(len(ind)):
        model = Sequential()
        model.add(Dense(nneur, input_dim=n2, activation=actf)) # one input neuron
        if ind[i] == 1:
            model.add(Dropout(0.2)) # Add dropout layer for specific trial
        model.add(Dense(nneur, activation=actf)) # repeat alpha number of times
        if ind[i] == 2:
            model.add(Dropout(0.2)) # Add dropout layer for specific trial
        model.add(Dense(nneur, activation=actf)) # repeat alpha number of times
        if ind[i] == 3:
            model.add(Dropout(0.2)) # Add dropout layer for specific trial
        model.add(Dense(nneur, activation=actf)) # repeat alpha number of times
        if ind[i] == 4:
            model.add(Dropout(0.2)) # Add dropout layer for specific trial
        model.add(Dense(nneur, activation=actf)) # repeat alpha number of times
        model.add(Dense(3)) # one output neuron
        opt = tf.keras.optimizers.legacy.SGD(learning_rate=alpha, momentum=0) # stochastic gradient descent optimizer
        model.compile(loss=mse, optimizer=opt, metrics=['mse', 'mae', 'mape'])
        # model.summary()
        history1 = model.fit(X2_tr, y_tr, epochs=epoch, validation_data=(X2_test, y_test), verbose=0, use_multiprocessing=-2) # validation for monitoring validation loss and metrics at the end of each epoch
        his.append(history1)
        tr_loss[i] = np.min(history1.history['loss']) # history1.history['loss'][-1]
        v_loss[i] = np.min(history1.history['val_loss']) #history1.history['val_loss'][-1]
        print(f'Completed test # {ind[i]} ---> val loss = {v_loss[i]}')
    
    # Training and Validation Heatmaps
    plt.figure(figsize=(12,5))
    plt.subplot(121)
    ax = sns.heatmap(tr_loss, annot=True, annot_kws={"fontsize":FS}, cbar_kws={'label': 'Training Loss'}, vmin=np.min(tr_loss), vmax=np.max(tr_loss))
    ax.set_yticklabels(ind, fontsize=FS*3/4);
    plt.title('Training Losses', fontsize=FS)
    plt.xlabel('Dropout Test Index', fontsize=FS)
    plt.ylabel('', fontsize=FS)
    plt.subplot(122)
    ax = sns.heatmap(v_loss, annot=True, annot_kws={"fontsize":FS}, cbar_kws={'label': 'Validation Loss'}, vmin=np.min(v_loss), vmax=np.max(v_loss))
    ax.set_yticklabels(ind, fontsize=FS*3/4);
    plt.title('Validation Losses', fontsize=FS)
    plt.xlabel('Dropout Test Index', fontsize=FS)
    plt.ylabel('', fontsize=FS)
    plt.savefig('G:\\My Drive\\RPI\\MANE 6962 Machine Learning\\Project\\Figures\\NNDropout1.png', dpi=300)
    
    # Training and Validation Epoch Plots
    ct = 0
    co = list(plt.rcParams['axes.prop_cycle'].by_key()['color']) + ['crimson', 'indigo', 'orange', 'red', 'blue', 'green', 'brown']
    plt.figure(figsize=(12, 7))
    for i in range(len(ind)):
        plt.plot(np.arange(1, epoch, 1), his[ct].history['loss'], '^--', linewidth=2, markersize=8, label=f'Training configuration {ind[i]}', color=co[ct])
        plt.plot(np.arange(1, epoch, 1), his[ct].history['val_loss'], 'v--', linewidth=2, markersize=8, label=f'Validation configuration {ind[i]}', color=co[ct])
        ct = ct+1
    plt.xlabel('Epoch', fontsize = FS)
    plt.ylabel('Loss', fontsize = FS)
    plt.title('Training/Testing Loss Comparison', fontsize = FS)
    plt.legend(loc='upper right', ncol=2)
    plt.savefig('G:\\My Drive\\RPI\\MANE 6962 Machine Learning\\Project\\Figures\\NNDropout2.png', dpi=300)
