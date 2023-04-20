# John Wylie, RIN# 661262436
# This script tunes hyperparameters for the linear regression in dataRegressor.py

import sys
import time
import numpy as np
import pandas as pd
# import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
# from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers, regularizers

## Starting Parameters
t1 = time.time()
pp_plt = False # boolean to determine whether to plot pairplot figure
nn1 = False # boolean: no. of neurons/no. of layers
nn2 = True # boolean: learning rate/no. or epochs
nn3 = False # boolean: dropout layers
nn4 = False # boolean: refined architecture/regularization
nn5 = False # boolean: 
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
dfX2 = pd.concat([dfX1[dfX1.columns[:4]], dfX1[dfX1.columns[-4:]]], axis=1)
X2 = pd.DataFrame.to_numpy(dfX2) # make into numpy array
# We don't need to repeat for y b/c it is the same for both full and abridged

# Abridged Dataset with Pressure Ports Only
dfX3 = dfX1[dfX1.columns[4:-4]]
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

if np.isnan(np.sum(X1)):
    sys.exit('X1 has NaN value')
if np.isnan(np.sum(X2)):
    sys.exit('X2 has NaN value')
if np.isnan(np.sum(X3)):
    sys.exit('X3 has NaN value')

# Check that all data falls in the range of [0,1]
# =============================================================================
# plt.figure()
# for i in range(n1):
#     plt.plot(X1[:,i],'--')
#     plt.title('Scaled X1 Data')
#     plt.xlabel('Data Index')
#     plt.ylabel('Feature Value')
# plt.figure()
# for i in range(n2):
#     plt.plot(X2[:,i],'--')
#     plt.title('Scaled X2 Data')
#     plt.xlabel('Data Index')
#     plt.ylabel('Feature Value')
# plt.figure()
# for i in range(n3):
#     plt.plot(X3[:,i],'--')
#     plt.title('Scaled X3 Data')
#     plt.xlabel('Data Index')
#     plt.ylabel('Feature Value')
# plt.figure()
# for i in range(p):
#     plt.plot(y[:,i],'--')
#     plt.title('Scaled y Data')
#     plt.xlabel('Data Index')
#     plt.ylabel('Feature Value')
# =============================================================================

## Visualize the Data
if pp_plt:
    sns.pairplot(dfX2, kind='scatter')

## Split Data into Training and Testing Splits
rs = 47
X1_tr, X1_test, y_tr, y_test = train_test_split(X1, y, test_size=0.3, random_state=rs)
X2_tr, X2_test, y_tr, y_test = train_test_split(X2, y, test_size=0.3, random_state=rs)
X3_tr, X3_test, y_tr, y_test = train_test_split(X3, y, test_size=0.3, random_state=rs)

actf = 'relu'
if nn1: # #neurons/#layers
    tr_history = []
    his = []
    nneur = np.array([400, 600, 1000]) # number of neurons per layer (barring output layer)
    nlayers = np.array([6, 8, 10]) # number of hidden layers
    # nneur = np.array([6, 7, 8]) # number of neurons per layer (barring output layer)
    # nlayers = np.array([1, 2]) # number of hidden layers
    epo = 300 # number of epochs
    # epo = 2 # number of epochs
    alpha = 0.001 # learning rate for optimizer
    opt = tf.keras.optimizers.legacy.SGD(learning_rate=alpha, momentum=0) # stochastic gradient descent optimizer
    mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM) # mean squared error for loss calculation
    
    tr_loss = np.zeros([len(nneur), len(nlayers)])
    v_loss = np.zeros([len(nneur), len(nlayers)])
    y_pred2 = []
    y_pred2sub = []
    for i in range(len(nneur)):
        y_pred2sub = []
        for j in range(len(nlayers)):
            model = Sequential()
            model.add(Dense(nneur[i], input_dim=n2, activation=actf)) # input layer
            for k in range(nlayers[j]):
                model.add(Dense(nneur[i], activation=actf)) # repeat nlayers number of times
            model.add(Dense(3)) # one output neuron
            model.compile(loss=mse, optimizer=opt, metrics=['mse', 'mae', 'mape'])
            model.summary()
            history1 = model.fit(X2_tr, y_tr, epochs=epo, validation_data=(X2_test, y_test), verbose=0, use_multiprocessing=-3) # validation for monitoring validation loss and metrics at the end of each epoch
            history1.history['nneur'] = nneur[i]
            history1.history['nlayers'] = nlayers[j]
            his.append(history1)
            tr_loss[i,j] = np.min(history1.history['loss']) # history1.history['loss'][-1]
            v_loss[i,j] = np.min(history1.history['val_loss']) #history1.history['val_loss'][-1]
            y_pred2sub.append(model.predict(X2_test, verbose=0, use_multiprocessing=-3))
            print(f'Completed: nneur = {nneur[i]}, nlayers = {nlayers[j]} ---> val loss = {v_loss[i,j]}')
        y_pred2.append(y_pred2sub)
        
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
    # plt.savefig('G:\\My Drive\\RPI\\MANE 6962 Machine Learning\\Project\\Figures\\NNnneurnlayers1v3.png', dpi=300)
    
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
    plt.title('Training/Validation Loss Comparison', fontsize = FS)
    plt.legend(loc='upper right', ncol=2)
    # plt.savefig('G:\\My Drive\\RPI\\MANE 6962 Machine Learning\\Project\\Figures\\NNnneurnlayers2v3.png', dpi=300)
    
    # Output Parameters
    ct = 1
    fig = plt.figure(figsize=(16, 10))
    for i in range(len(nneur)):
        for j in range(len(nlayers)):
            plt.subplot(len(nneur), len(nlayers), ct)
            plt.plot(y_test[:,0], y_test[:,1], 'ko', label='Test Data')
            plt.plot(y_pred2[i][j][:,0], y_pred2[i][j][:,1], '*', label='Predicted Data')
            plt.legend(loc='center right')
            plt.title(f'nNeurons: {nneur[i]}, nLayers: {nlayers[j]}, ')
            if j == 0:
                plt.ylabel(r'$C_l$ (scaled)')
            if i == len(nneur)-1:
                plt.xlabel(r'$C_d$ (scaled)')
            ct = ct+1
    fig.tight_layout()
    # plt.savefig('G:\\My Drive\\RPI\\MANE 6962 Machine Learning\\Project\\Figures\\NNnneurnlayers3v3.png', dpi=300)
    
if nn2: # Epoch/Learning Rate
    tr_history = []
    his = []
    nneur = 400
    epoch = np.array([200, 300, 400]) # number of epochs
    alpha = np.array([0.001, 0.002, 0.003, 0.004]) # learning rate
    # epoch = np.array([6, 7, 8]) # number of epochs
    # alpha = np.array([0.01, 0.02]) # learning rate
    mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM) # mean squared error for loss calculation
    
    tr_loss = np.zeros([len(epoch), len(alpha)])
    v_loss = np.zeros([len(epoch), len(alpha)])
    y_pred2 = []
    y_pred2sub = []
    for i in range(len(epoch)):
        y_pred2sub = []
        for j in range(len(alpha)):
            model = Sequential()
            model.add(Dense(nneur, input_dim=n2, activation=actf)) # input layer
            model.add(Dense(nneur, activation=actf)) # hidden layer
            model.add(Dense(nneur, activation=actf)) # hidden layer
            model.add(Dense(nneur, activation=actf)) # hidden layer
            model.add(Dense(nneur, activation=actf)) # hidden layer
            model.add(Dense(3)) # one output neuron
            opt = tf.keras.optimizers.legacy.SGD(learning_rate=alpha[j], momentum=0) # stochastic gradient descent optimizer
            model.compile(loss=mse, optimizer=opt, metrics=['mse', 'mae', 'mape'])
            model.summary()
            history1 = model.fit(X2_tr, y_tr, epochs=epoch[i], validation_data=(X2_test, y_test), verbose=0, use_multiprocessing=-2)
            history1.history['epoch'] = epoch[i]
            history1.history['alpha'] = alpha[j]
            his.append(history1)
            tr_loss[i,j] = np.min(history1.history['loss']) # history1.history['loss'][-1]
            v_loss[i,j] = np.min(history1.history['val_loss']) #history1.history['val_loss'][-1]
            y_pred2sub.append(model.predict(X2_test, verbose=0, use_multiprocessing=-2))
            print(f'Completed: epoch = {epoch[i]}, alpha = {alpha[j]} ---> val loss = {v_loss[i,j]}')
        y_pred2.append(y_pred2sub)
        
        
    # Training and Validation Heatmaps
    plt.figure(figsize=(12,5))
    plt.subplot(121)
    ax = sns.heatmap(tr_loss, annot=True, annot_kws={"fontsize":FS}, cbar_kws={'label': 'Training Loss'}, vmin=np.min(tr_loss), vmax=np.max(tr_loss))
    ax.set_xticklabels(alpha, fontsize=FS*3/4);
    ax.set_yticklabels(epoch, fontsize=FS*3/4);
    plt.title('Training Losses', fontsize=FS)
    plt.xlabel(r'Learning Rate ($\alpha$)', fontsize=FS)
    plt.ylabel('nEpochs', fontsize=FS)
    plt.subplot(122)
    ax = sns.heatmap(v_loss, annot=True, annot_kws={"fontsize":FS}, cbar_kws={'label': 'Validation Loss'}, vmin=np.min(v_loss), vmax=np.max(v_loss))
    ax.set_xticklabels(alpha, fontsize=FS*3/4);
    ax.set_yticklabels(epoch, fontsize=FS*3/4);
    plt.title('Validation Losses', fontsize=FS)
    plt.xlabel(r'Learning Rate ($\alpha$)', fontsize=FS)
    # plt.ylabel('nEpochs', fontsize=FS)
    # plt.savefig('G:\\My Drive\\RPI\\MANE 6962 Machine Learning\\Project\\Figures\\NNepochalpha1.png', dpi=300)
    
    # Training and Validation Epoch Plots
    ct = 0
    co = list(plt.rcParams['axes.prop_cycle'].by_key()['color']) + ['crimson', 'indigo', 'orange', 'red', 'blue', 'green', 'brown']
    plt.figure(figsize=(12, 7))
    for i in range(len(epoch)):
        for j in range(len(alpha)):
            plt.plot(np.arange(his[ct].history['epoch']), his[ct].history['loss'], '^--', linewidth=2, markersize=8, label=rf'Training epochs: {epoch[i]}, $\alpha$: {alpha[j]}', color=co[ct])
            plt.plot(np.arange(his[ct].history['epoch']), his[ct].history['val_loss'], 'v--', linewidth=2, markersize=8, label=rf'Validation epochs: {epoch[i]}, $\alpha$: {alpha[j]}', color=co[ct])
            ct = ct+1
    plt.xlabel('Epoch', fontsize = FS)
    plt.ylabel('Loss', fontsize = FS)
    # plt.ylim([0, 1.2*np.max(tr_history1.history['val_loss'])])
    plt.title('Training/Validation Loss Comparison', fontsize = FS)
    plt.legend(loc='upper right', ncol=2)
    # plt.savefig('G:\\My Drive\\RPI\\MANE 6962 Machine Learning\\Project\\Figures\\NNepochalpha2.png', dpi=300)
    
    # Output Parameters
    ct = 1
    fig = plt.figure(figsize=(16, 10))
    for i in range(len(epoch)):
        for j in range(len(alpha)):
            plt.subplot(len(epoch), len(alpha), ct)
            plt.plot(y_test[:,0], y_test[:,1], 'ko', label='Test Data')
            plt.plot(y_pred2[i][j][:,0], y_pred2[i][j][:,1], '*', label='Predicted Data')
            plt.legend(loc='center right')
            plt.title(rf'epochs: {epoch[i]}, $\alpha$: {alpha[j]}, ')
            if j == 0:
                plt.ylabel(r'$C_l$ (scaled)')
            if i == len(epoch)-1:
                plt.xlabel(r'$C_d$ (scaled)')
            ct = ct+1
    fig.tight_layout()
    # plt.savefig('G:\\My Drive\\RPI\\MANE 6962 Machine Learning\\Project\\Figures\\NNepochalpha3.png', dpi=300)

if nn3: # Dropout Layer 
    tr_history = []
    his = []
    ind = np.array([0, 1, 2, 3, 4, 5]) # dropout layer position
    alpha = 0.004 # learning rate
    epoch = 300 # number of epochs
    # epoch = 3
    nneur = 400 # number of neurons per input/hidden layer
    # nneur = 20
    mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM) # mean squared error for loss calculation
    
    tr_loss = np.zeros([len(ind),1])
    v_loss = np.zeros([len(ind),1])
    y_pred2 = []
    for i in range(len(ind)):
        model = Sequential()
        model.add(Dense(nneur, input_dim=n2, activation=actf)) # input layer
        if ind[i] == 1:
            model.add(Dropout(0.2)) # Add dropout layer for specific trial
        model.add(Dense(nneur, activation=actf))
        if ind[i] == 2:
            model.add(Dropout(0.2)) # Add dropout layer for specific trial
        model.add(Dense(nneur, activation=actf))
        if ind[i] == 3:
            model.add(Dropout(0.2)) # Add dropout layer for specific trial
        model.add(Dense(nneur, activation=actf))
        if ind[i] == 4:
            model.add(Dropout(0.2)) # Add dropout layer for specific trial
        model.add(Dense(nneur, activation=actf))
        if ind[i] == 5:
            model.add(Dropout(0.2)) # Add dropout layer for specific trial
        model.add(Dense(3)) # output layer
        opt = tf.keras.optimizers.legacy.SGD(learning_rate=alpha, momentum=0) # stochastic gradient descent optimizer
        model.compile(loss=mse, optimizer=opt, metrics=['mse', 'mae', 'mape'])
        model.summary()
        history1 = model.fit(X2_tr, y_tr, epochs=epoch, validation_data=(X2_test, y_test), verbose=0, use_multiprocessing=-2) # validation for monitoring validation loss and metrics at the end of each epoch
        his.append(history1)
        tr_loss[i] = np.min(history1.history['loss']) # history1.history['loss'][-1]
        v_loss[i] = np.min(history1.history['val_loss']) #history1.history['val_loss'][-1]
        y_pred2.append(model.predict(X2_test, verbose=0, use_multiprocessing=-3))
        print(f'Completed test # {ind[i]} ---> val loss = {v_loss[i]}')
    
    # Training and Validation Heatmaps
    plt.figure(figsize=(12,5))
    plt.subplot(121)
    ax = sns.heatmap(tr_loss, annot=True, annot_kws={"fontsize":FS}, cbar_kws={'label': 'Training Loss'}, vmin=np.min(tr_loss), vmax=np.max(tr_loss))
    ax.set_yticklabels(ind, fontsize=FS*3/4);
    plt.title('Training Losses', fontsize=FS)
    plt.xlabel('', fontsize=FS)
    plt.ylabel('Dropout Test Index', fontsize=FS)
    plt.subplot(122)
    ax = sns.heatmap(v_loss, annot=True, annot_kws={"fontsize":FS}, cbar_kws={'label': 'Validation Loss'}, vmin=np.min(v_loss), vmax=np.max(v_loss))
    ax.set_yticklabels(ind, fontsize=FS*3/4);
    plt.title('Validation Losses', fontsize=FS)
    plt.xlabel('', fontsize=FS)
    # plt.ylabel('Dropout Test index', fontsize=FS)
    plt.savefig('G:\\My Drive\\RPI\\MANE 6962 Machine Learning\\Project\\Figures\\NNDropout1.png', dpi=300)
    
    # Training and Validation Epoch Plots
    ct = 0
    co = list(plt.rcParams['axes.prop_cycle'].by_key()['color']) + ['crimson', 'indigo', 'orange', 'red', 'blue', 'green', 'brown']
    plt.figure(figsize=(12, 7))
    for i in range(len(ind)):
        plt.plot(np.arange(0, epoch, 1)+1, his[ct].history['loss'], '^--', linewidth=2, markersize=8, label=f'Training: dropout after layer {ind[i]+1}', color=co[ct])
        plt.plot(np.arange(0, epoch, 1)+1, his[ct].history['val_loss'], 'v--', linewidth=2, markersize=8, label=f'Validation: dropout after layer {ind[i]+1}', color=co[ct])
        ct = ct+1
    plt.xlabel('Epoch', fontsize = FS)
    plt.ylabel('Loss', fontsize = FS)
    plt.title('Training/Validation Loss Comparison', fontsize = FS)
    plt.legend(loc='upper right', ncol=2)
    plt.savefig('G:\\My Drive\\RPI\\MANE 6962 Machine Learning\\Project\\Figures\\NNDropout2.png', dpi=300)
    
    # Output Parameters
    fig = plt.figure(figsize=(5, 12))
    for i in range(len(ind)):
        plt.subplot(len(ind), 1, i+1)
        plt.plot(y_test[:,0], y_test[:,1], 'ko', label='Test Data')
        plt.plot(y_pred2[i][:,0], y_pred2[i][:,1], '*', label='Predicted Data')
        plt.legend(loc='center right')
        plt.title(f'Dropout after Layer {ind[i]+1}')
        plt.ylabel(r'$C_l$ (scaled)')
        if i == len(ind)-1:
            plt.xlabel(r'$C_d$ (scaled)')
    fig.tight_layout()
    plt.savefig('G:\\My Drive\\RPI\\MANE 6962 Machine Learning\\Project\\Figures\\NNDropout3.png', dpi=300)

if nn4: # Regularization
    tr_history = []
    his = []
    reg = [0, regularizers.L1(1e-4), regularizers.L2(1e-4), regularizers.L1L2(1e-4)] # regularizers
    regstr = ['None', 'L1', 'L2', 'L1L2']
    alpha = 0.004 # learning rate
    epoch = 300 # number of epochs
    # epoch = 3
    nneur = 200 # number of neurons per input/hidden layer
    # nneur = 20
    mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM) # mean squared error for loss calculation
    
    tr_loss = np.zeros([len(reg),1])
    v_loss = np.zeros([len(reg),1])
    y_pred2 = []
    for i in range(len(reg)):
        model = Sequential()
        model.add(Dense(nneur, input_dim=n2, activation=actf)) # input layer
        for k in range(4):
            if i != 0: # Do not regularize for baseline
                model.add(Dense(nneur, activation=actf, kernel_regularizer=reg[i]))
        model.add(Dense(3)) # one output neuron
        opt = tf.keras.optimizers.legacy.SGD(learning_rate=alpha, momentum=0) # stochastic gradient descent optimizer
        model.compile(loss=mse, optimizer=opt, metrics=['mse', 'mae', 'mape'])
        model.summary()
        history1 = model.fit(X2_tr, y_tr, epochs=epoch, validation_data=(X2_test, y_test), verbose=0, use_multiprocessing=-2) # validation for monitoring validation loss and metrics at the end of each epoch
        his.append(history1)
        tr_loss[i] = np.min(history1.history['loss']) # history1.history['loss'][-1]
        v_loss[i] = np.min(history1.history['val_loss']) #history1.history['val_loss'][-1]
        y_pred2.append(model.predict(X2_test, verbose=0, use_multiprocessing=-3))
        print(f'Completed test with regularizer {regstr[i]} ---> val loss = {v_loss[i]}')
        
    # Training and Validation Heatmaps
    plt.figure(figsize=(9,5))
    plt.subplot(121)
    ax = sns.heatmap(tr_loss, annot=True, annot_kws={"fontsize":FS}, cbar_kws={'label': 'Training Loss'}, vmin=np.min(tr_loss), vmax=np.max(tr_loss))
    ax.set_yticklabels(regstr, fontsize=FS*3/4);
    plt.title('Training Losses', fontsize=FS)
    plt.xlabel('Architecture Configuration', fontsize=FS)
    plt.ylabel('Regularization', fontsize=FS)
    plt.subplot(122)
    ax = sns.heatmap(v_loss, annot=True, annot_kws={"fontsize":FS}, cbar_kws={'label': 'Validation Loss'}, vmin=np.min(v_loss), vmax=np.max(v_loss))
    ax.set_yticklabels(regstr, fontsize=FS*3/4);
    plt.title('Validation Losses', fontsize=FS)
    plt.xlabel('Architecture Configuration', fontsize=FS)
    # plt.ylabel('Regularization', fontsize=FS)
    plt.savefig('G:\\My Drive\\RPI\\MANE 6962 Machine Learning\\Project\\Figures\\NNReg1.png', dpi=300)
    
    # Training and Validation Epoch Plots
    ct = 0
    co = list(plt.rcParams['axes.prop_cycle'].by_key()['color']) + ['crimson', 'indigo', 'orange', 'red', 'blue', 'green', 'brown']
    plt.figure(figsize=(12, 7))
    for i in range(len(reg)):
            plt.plot(np.arange(0, epoch, 1)+1, his[ct].history['loss'], '^--', linewidth=2, markersize=8, label=f'Training: reg. {regstr[i]}', color=co[ct])
            plt.plot(np.arange(0, epoch, 1)+1, his[ct].history['val_loss'], 'v--', linewidth=2, markersize=8, label=f'Validation: reg. {regstr[i]}', color=co[ct])
            ct = ct+1
    plt.xlabel('Epoch', fontsize = FS)
    plt.ylabel('Loss', fontsize = FS)
    plt.title('Training/Validation Loss Comparison', fontsize = FS)
    plt.legend(loc='upper right', ncol=2, fontsize = FS)
    plt.savefig('G:\\My Drive\\RPI\\MANE 6962 Machine Learning\\Project\\Figures\\NNReg2.png', dpi=300)    
    
    # Output Parameters
    fig = plt.figure(figsize=(5, 12))
    for i in range(len(reg)):
        plt.subplot(len(reg), 1, i+1)
        plt.plot(y_test[:,0], y_test[:,1], 'ko', label='Test Data')
        plt.plot(y_pred2[i][:,0], y_pred2[i][:,1], '*', label='Predicted Data')
        plt.legend(loc='center right', fontsize = FS)
        plt.title(f'Regularization: {regstr[i]}', fontsize = FS)
        plt.ylabel(r'$C_l$ (scaled)', fontsize = FS)
        if i == len(reg)-1:
            plt.xlabel(r'$C_d$ (scaled)', fontsize = FS)
    fig.tight_layout()
    plt.savefig('G:\\My Drive\\RPI\\MANE 6962 Machine Learning\\Project\\Figures\\NNReg3.png', dpi=300)
    
if nn5: # Activation Functions and Optimizers
    tr_history = []
    his = []
    alpha = 0.03 # learning rate
    opt1 = tf.keras.optimizers.legacy.SGD(learning_rate=alpha, momentum=0) # stochastic gradient descent optimizer
    opt2 = tf.keras.optimizers.legacy.Adam(learning_rate=alpha) # adam
    opt3 = tf.keras.optimizers.legacy.RMSprop(learning_rate=alpha, momentum=0)
    opt = [opt1, opt2, opt3] # optimizers
    optstr = ['SGD', 'Adam', 'RMSprop']
    act = ['relu', 'sigmoid', 'tanh', 'selu', 'exponential']
    epoch = 300 # number of epochs
    # epoch = 3
    nneur = 400 # number of neurons per input/hidden layer
    # nneur = 20
    mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM) # mean squared error for loss calculation
    
    tr_loss = np.zeros([len(act),len(opt)])
    v_loss = np.zeros([len(act),len(opt)])
    y_pred2 = []
    for i in range(len(act)):
        y_pred2sub = []
        for j in range(len(opt)):
            model = Sequential()
            model.add(Dense(nneur, input_dim=n2, activation=act[i])) # input layer
            model.add(Dense(nneur, activation=act[i], kernel_regularizer=regularizers.L1(1e-4)))
            model.add(Dense(nneur, activation=act[i], kernel_regularizer=regularizers.L1(1e-4)))
            model.add(Dense(nneur, activation=act[i], kernel_regularizer=regularizers.L1(1e-4)))
            model.add(Dense(nneur, activation=act[i], kernel_regularizer=regularizers.L1(1e-4)))
            model.add(Dense(3)) # one output neuron
            model.compile(loss=mse, optimizer=opt[j], metrics=['mse', 'mae', 'mape'])
            model.summary()
            history1 = model.fit(X2_tr, y_tr, epochs=epoch, validation_data=(X2_test, y_test), verbose=0, use_multiprocessing=-2) # validation for monitoring validation loss and metrics at the end of each epoch
            his.append(history1)
            tr_loss[i,j] = np.min(history1.history['loss']) # history1.history['loss'][-1]
            v_loss[i,j] = np.min(history1.history['val_loss']) #history1.history['val_loss'][-1]
            y_pred2sub.append(model.predict(X2_test, verbose=0, use_multiprocessing=-3))
            print(f'Completed test with {act[i]} and {optstr[j]} ---> val loss = {v_loss[i,j]}')
        y_pred2.append(y_pred2sub)
        
    # Training and Validation Heatmaps
    plt.figure(figsize=(12,5))
    plt.subplot(121)
    ax = sns.heatmap(tr_loss, annot=True, annot_kws={"fontsize":FS}, cbar_kws={'label': 'Training Loss'}, vmin=0, vmax=np.max(tr_loss)) # hard code 0 b/c NaNs
    ax.set_xticklabels(optstr, fontsize=FS*3/4);
    ax.set_yticklabels(act, fontsize=FS*3/4);
    plt.title('Training Losses', fontsize=FS)
    plt.xlabel('Activation Function', fontsize=FS)
    plt.ylabel('Optimizer', fontsize=FS)
    plt.subplot(122)
    ax = sns.heatmap(v_loss, annot=True, annot_kws={"fontsize":FS}, cbar_kws={'label': 'Validation Loss'}, vmin=0, vmax=np.max(v_loss)) # hard code 0 b/c NaNs
    ax.set_xticklabels(optstr, fontsize=FS*3/4);
    ax.set_yticklabels(act, fontsize=FS*3/4);
    plt.title('Validation Losses', fontsize=FS)
    plt.xlabel('Activation Function', fontsize=FS)
    # plt.ylabel('Optularization', fontsize=FS)
    plt.savefig('G:\\My Drive\\RPI\\MANE 6962 Machine Learning\\Project\\Figures\\NNactopt1.png', dpi=300)
    
    # Training and Validation Epoch Plots
    ct = 0
    co = list(plt.rcParams['axes.prop_cycle'].by_key()['color']) + ['crimson', 'indigo', 'orange', 'red', 'blue', 'green', 'brown']
    plt.figure(figsize=(12, 7))
    for i in range(len(act)):
        for j in range(len(opt)):
            if not np.isnan(tr_loss[i,j]):
                plt.plot(np.arange(0, epoch, 1)+1, his[ct].history['loss'], '^--', linewidth=2, markersize=8, label=f'Training: {act[i]}, {optstr[j]}', color=co[ct])
            if not np.isnan(v_loss[i,j]):
                plt.plot(np.arange(0, epoch, 1)+1, his[ct].history['val_loss'], 'v--', linewidth=2, markersize=8, label=f'Validation: {act[i]}, {optstr[j]}', color=co[ct])
            ct = ct+1
    plt.xlabel('Epoch', fontsize = FS)
    plt.ylabel('Loss', fontsize = FS)
    plt.ylim((0, 50))
    plt.title('Training/Validation Loss Comparison', fontsize = FS)
    plt.legend(loc='upper right', ncol=2)
    plt.savefig('G:\\My Drive\\RPI\\MANE 6962 Machine Learning\\Project\\Figures\\NNactopt2.png', dpi=300)    

    # Output Parameters
    ct = 1
    fig = plt.figure(figsize=(10, 16))
    for i in range(len(act)):
        for j in range(len(opt)):
            plt.subplot(len(act), len(opt), ct)
            plt.plot(y_test[:,0], y_test[:,1], 'ko', label='Test Data')
            plt.plot(y_pred2[i][j][:,0], y_pred2[i][j][:,1], '*', label='Predicted Data')
            plt.legend(loc='center right')
            plt.title(f'{act[i]}, {optstr[j]}', fontsize = FS)
            if j == 0:
                plt.ylabel(r'$C_l$ (scaled)', fontsize = FS)
            if i == len(act)-1:
                plt.xlabel(r'$C_d$ (scaled)', fontsize = FS)
            ct = ct+1
            plt.legend(loc='upper right', ncol=2)
    fig.tight_layout()
    plt.savefig('G:\\My Drive\\RPI\\MANE 6962 Machine Learning\\Project\\Figures\\NNactopt3.png', dpi=300)
    
## Unused:
# =============================================================================
# if nn4: # Refined Architecture and Regularization
#     tr_history = []
#     his = []
#     reg = np.array([0, regularizers.L1(1e-4), regularizers.L2(1e-4), regularizers.L1L2(1e-4)]) # regularizers
#     arch = []
#     arch.append(np.array([200, 200, 200, 200]))
#     arch.append(np.array([512, 256, 32]))
#     arch.append(np.array([256, 256, 128, 128, 32]))
#     alpha = 0.03 # learning rate
#     epoch = 300 # number of epochs
#     epoch = 3
#     nneur = 200 # number of neurons per input/hidden layer
#     nneur = 20
#     mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM) # mean squared error for loss calculation
#     
#     tr_loss = np.zeros([len(reg),len(arch)])
#     v_loss = np.zeros([len(reg),len(arch)])
#     for i in range(len(reg)):
#         for j in range(len(arch)):
#             model = Sequential()
#             model.add(Dense(nneur, input_dim=n2, activation=actf)) # input layer
#             for k in range(len(arch[j])):
#                 if i > 1:
#                     model.add(Dense(arch[j][k], activation=actf, kernel_regularizer=reg[i]))
#             model.add(Dense(3)) # one output neuron
#             opt = tf.keras.optimizers.legacy.SGD(learning_rate=alpha, momentum=0) # stochastic gradient descent optimizer
#             model.compile(loss=mse, optimizer=opt, metrics=['mse', 'mae', 'mape'])
#             model.summary()
#             history1 = model.fit(X2_tr, y_tr, epochs=epoch, validation_data=(X2_test, y_test), verbose=0, use_multiprocessing=-2) # validation for monitoring validation loss and metrics at the end of each epoch
#             his.append(history1)
#             tr_loss[i,j] = np.min(history1.history['loss']) # history1.history['loss'][-1]
#             v_loss[i,j] = np.min(history1.history['val_loss']) #history1.history['val_loss'][-1]
#             print(f'Completed test with regularizer {reg[i]} and configuration {j} ---> val loss = {v_loss[i,j]}')
#         
#     # Training and Validation Heatmaps
#     plt.figure(figsize=(12,5))
#     plt.subplot(121)
#     ax = sns.heatmap(tr_loss, annot=True, annot_kws={"fontsize":FS}, cbar_kws={'label': 'Training Loss'}, vmin=np.min(tr_loss), vmax=np.max(tr_loss))
#     ax.set_xticklabels([0,1,2], fontsize=FS*3/4);
#     ax.set_yticklabels(['None','L1','L2','L1L2'], fontsize=FS*3/4);
#     plt.title('Training Losses', fontsize=FS)
#     plt.xlabel('Architecture Configuration', fontsize=FS)
#     plt.ylabel('Regularization', fontsize=FS)
#     plt.subplot(122)
#     ax = sns.heatmap(v_loss, annot=True, annot_kws={"fontsize":FS}, cbar_kws={'label': 'Validation Loss'}, vmin=np.min(v_loss), vmax=np.max(v_loss))
#     ax.set_xticklabels([0,1,2], fontsize=FS*3/4);
#     ax.set_yticklabels(['None','L1','L2','L1L2'], fontsize=FS*3/4);
#     plt.title('Validation Losses', fontsize=FS)
#     plt.xlabel('Architecture Configuration', fontsize=FS)
#     # plt.ylabel('Regularization', fontsize=FS)
#     plt.savefig('G:\\My Drive\\RPI\\MANE 6962 Machine Learning\\Project\\Figures\\NNRegArch1.png', dpi=300)
#     
#     # Training and Validation Epoch Plots
#     ct = 0
#     co = list(plt.rcParams['axes.prop_cycle'].by_key()['color']) + ['crimson', 'indigo', 'orange', 'red', 'blue', 'green', 'brown']
#     plt.figure(figsize=(12, 7))
#     for i in range(len(reg)):
#         for j in range(len(arch)):
#             plt.plot(np.arange(0, epoch, 1)+1, his[ct].history['loss'], '^--', linewidth=2, markersize=8, label=f'Training config. {j}, reg. {reg[i]}', color=co[ct])
#             plt.plot(np.arange(0, epoch, 1)+1, his[ct].history['val_loss'], 'v--', linewidth=2, markersize=8, label=f'Validation config. {j}, reg. {reg[i]}', color=co[ct])
#             ct = ct+1
#     plt.xlabel('Epoch', fontsize = FS)
#     plt.ylabel('Loss', fontsize = FS)
#     plt.title('Training/Validation Loss Comparison', fontsize = FS)
#     plt.legend(loc='upper right', ncol=2)
#     plt.savefig('G:\\My Drive\\RPI\\MANE 6962 Machine Learning\\Project\\Figures\\NNRegArch2.png', dpi=300)    
#     
#     y_pred2 = model.predict(X2_test, verbose=0, use_multiprocessing=-3)
#     plt.figure(figsize=(8, 5))
#     plt.plot(y_test[:,0], y_test[:,1], 'ko', label='Test Data')
#     plt.plot(y_pred2[:,0], y_pred2[:,1], 'r*', label='Predictions')
# =============================================================================

print(f'Duration: {time.time()-t1} seconds')