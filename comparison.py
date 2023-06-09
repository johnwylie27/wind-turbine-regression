# John Wylie, RIN# 661262436
# This script runs the final setup and generates the figures to show the results.

import sys
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers, regularizers

## Starting Parameters
t1 = time.time()
pp_plt = 0 # boolean to determine whether to plot pairplot figure
nn1 = False # boolean to determine whether to run the NN on the full data
nn2 = True # boolean to determine whether to run the NN on the partial data
nn3 = True # boolean to determine whether to run the NN on the pressure port data only
FS = 15 # font size for plotting labels

#%% Load Data and PreProcessing
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

## Split Data into Training and Testing Splits
rs = 49
X1_tr, X1_test, y_tr, y_test = train_test_split(X1, y, test_size=0.3, random_state=rs)
X2_tr, X2_test, y_tr, y_test = train_test_split(X2, y, test_size=0.3, random_state=rs)
X3_tr, X3_test, y_tr, y_test = train_test_split(X3, y, test_size=0.3, random_state=rs)

#%% X2
actf = 'relu'
tr_history = []
his = []
nneur = 1000
epoch = 1000 # number of epochs
alpha = 0.004 # learning rate
reg = regularizers.L2(1e-4)
# epoch = np.array([6, 7, 8]) # number of epochs
# alpha = np.array([0.01, 0.02]) # learning rate
mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM) # mean squared error for loss calculation

# tr_loss = np.zeros([len(epoch), len(alpha)])
# v_loss = np.zeros([len(epoch), len(alpha)])
model = Sequential()
model.add(Dense(nneur, input_dim=n2, activation=actf, kernel_regularizer=reg)) # input layer
model.add(Dense(nneur, activation=actf, kernel_regularizer=reg)) # hidden layer
model.add(Dense(nneur, activation=actf, kernel_regularizer=reg)) # hidden layer
model.add(Dense(nneur, activation=actf, kernel_regularizer=reg)) # hidden layer
model.add(Dense(nneur, activation=actf, kernel_regularizer=reg)) # hidden layer
model.add(Dense(nneur, activation=actf, kernel_regularizer=reg)) # hidden layer
model.add(Dropout(0.2)) # Add dropout layer
model.add(Dense(nneur, activation=actf, kernel_regularizer=reg)) # hidden layer
model.add(Dense(nneur, activation=actf, kernel_regularizer=reg)) # hidden layer
model.add(Dense(nneur, activation=actf, kernel_regularizer=reg)) # hidden layer
model.add(Dense(nneur, activation=actf, kernel_regularizer=reg)) # hidden layer
model.add(Dense(nneur, activation=actf, kernel_regularizer=reg)) # hidden layer
model.add(Dense(3)) # one output neuron
opt = tf.keras.optimizers.legacy.SGD(learning_rate=alpha, momentum=0) # stochastic gradient descent optimizer
model.compile(loss=mse, optimizer=opt, metrics=['mse'])
model.summary()
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50)
history1 = model.fit(X2_tr, y_tr, epochs=epoch, validation_data=(X2_test, y_test), callbacks=callback, verbose=0, use_multiprocessing=-2)
history1.history['epoch'] = epoch
history1.history['alpha'] = alpha
his.append(history1)
tr_loss = np.min(history1.history['loss']) # history1.history['loss'][-1]
v_loss = np.min(history1.history['val_loss']) #history1.history['val_loss'][-1]
y_pred2 = model.predict(X2_test, verbose=0, use_multiprocessing=-2)
print('Completed NN for X2')
print(f'Duration: {time.time()-t1} seconds')

#%% X3
## Pressure Data
nneur = 400
epoch = 400 # number of epochs
alpha = 0.004 # learning rate

model = Sequential()
model.add(Dense(nneur, input_dim=n3, activation=actf, kernel_regularizer=reg)) # input layer
model.add(Dense(nneur, activation=actf, kernel_regularizer=reg)) # repeat alpha number of times
model.add(Dense(nneur, activation=actf, kernel_regularizer=reg)) # repeat alpha number of times
model.add(Dense(nneur, activation=actf, kernel_regularizer=reg)) # repeat alpha number of times
model.add(Dense(nneur, activation=actf, kernel_regularizer=reg)) # repeat alpha number of times
model.add(Dense(3)) # output layer
opt = tf.keras.optimizers.legacy.SGD(learning_rate=alpha, momentum=0) # stochastic gradient descent optimizer
mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM) # mean squared error for loss calculation
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50)
model.compile(loss=mse, optimizer=opt, metrics=['mse', 'mae', 'mape'])
tr_history3 = model.fit(X3_tr, y_tr, epochs=epoch, validation_data=(X3_test, y_test), callbacks=[callback], verbose=0, use_multiprocessing=-3)
model.summary()
y_pred3 = model.predict(X3_test, verbose=0, use_multiprocessing=-2)
test_loss3 = mean_squared_error(y_test, y_pred3)
print(test_loss3)


#%% Rescale data back from [0,1] to original scaling
for i in range(p):
    y_pred2[:,i] = (y_pred2[:,i]*(scaley[i,1] - scaley[i,0])) + scaley[i,0]
    y_pred3[:,i] = (y_pred3[:,i]*(scaley[i,1] - scaley[i,0])) + scaley[i,0]
    y_test[:,i] = (y_test[:,i]*(scaley[i,1] - scaley[i,0])) + scaley[i,0]
    
#%% Plotting
# NN for X2
# Training and Validation Epoch Plots
co = list(plt.rcParams['axes.prop_cycle'].by_key()['color']) + ['crimson', 'indigo', 'orange', 'red', 'blue', 'green', 'brown']
fig = plt.figure(figsize=(5, 5))
plt.plot(np.arange(len(his[0].history['loss'])), his[0].history['loss'], 'b^--', linewidth=2, markersize=4, label='Training')
plt.plot(np.arange(len(his[0].history['loss'])), his[0].history['val_loss'], 'rv--', linewidth=2, markersize=4, label='Validation')
plt.xlabel('Epoch', fontsize = FS)
plt.ylabel(r'$\Sigma MSE$', fontsize = FS)
plt.ylim([0, 1.2*np.max(his[0].history['val_loss'])])
plt.title('Test Paramete', fontsize = FS)
plt.legend(loc='upper right', fontsize = FS)
fig.tight_layout()
# plt.savefig('G:\\My Drive\\RPI\\MANE 6962 Machine Learning\\Project\\Figures\\NNFinalX2_1.png', dpi=300)

fig = plt.figure(figsize=(5, 5))
plt.plot(np.arange(len(his[0].history['mse'])), his[0].history['mse'], 'b^--', linewidth=2, markersize=4, label=('Training'))
plt.plot(np.arange(len(his[0].history['mse'])), his[0].history['val_mse'], 'rv--', linewidth=2, markersize=4, label=('Validation'))
plt.xlabel('Epoch', fontsize = FS)
plt.ylabel('MSE', fontsize = FS)
plt.ylim([0, 1.2*np.max(his[0].history['val_mse'])])
plt.title('Test Parameters', fontsize = FS)
plt.legend(loc='upper right', fontsize = FS)
fig.tight_layout()
# plt.savefig('G:\\My Drive\\RPI\\MANE 6962 Machine Learning\\Project\\Figures\\X2_mse.png', dpi=300)

# Output Parameters
fig = plt.figure(figsize=(5, 5))
plt.plot(y_test[:,0], y_test[:,1], 'ko', label='Test Data')
plt.plot(y_pred2[:,0], y_pred2[:,1], 'g*', label='Test Parameters')
plt.plot(y_pred3[:,0], y_pred3[:,1], 'rx', label='Pressure Data')
plt.legend(loc='center right', fontsize = FS)
plt.ylabel(r'$C_l$', fontsize = FS)
plt.xlabel(r'$C_d$', fontsize = FS)
plt.xlim((-0.05, 0.35))
plt.ylim((-1.10, 1.4))
fig.tight_layout()
plt.savefig('G:\\My Drive\\RPI\\MANE 6962 Machine Learning\\Project\\Figures\\NNFinalX2_2new.png', dpi=300)

fig = plt.figure(figsize=(5, 5))
plt.plot(y_test[:,0], y_test[:,2], 'ko', label='Test Data')
plt.plot(y_pred2[:,0], y_pred2[:,2], 'g*', label='Test Parameters')
plt.plot(y_pred3[:,0], y_pred3[:,2], 'rx', label='Pressure Data')
plt.legend(loc='upper left', fontsize = FS)
plt.ylabel(r'$C_m$', fontsize = FS)
plt.xlabel(r'$C_d$', fontsize = FS)
# plt.xlim((-0.05, 0.35))
# plt.ylim((-1.10, 1.4))
fig.tight_layout()
plt.savefig('G:\\My Drive\\RPI\\MANE 6962 Machine Learning\\Project\\Figures\\NNFinalX2CdCm.png', dpi=300)

fig = plt.figure(figsize=(5, 5))
plt.plot(y_test[:,1], y_test[:,2], 'ko', label='Test Data')
plt.plot(y_pred2[:,1], y_pred2[:,2], 'g*', label='Test Parameters')
plt.plot(y_pred3[:,1], y_pred3[:,2], 'rx', label='Pressure Data')
plt.legend(loc='center right', fontsize = FS)
plt.ylabel(r'$C_m$', fontsize = FS)
plt.xlabel(r'$C_l$', fontsize = FS)
# plt.xlim((-0.05, 0.35))
# plt.ylim((-1.10, 1.4))
fig.tight_layout()
plt.savefig('G:\\My Drive\\RPI\\MANE 6962 Machine Learning\\Project\\Figures\\NNFinalX2ClCm.png', dpi=300)

# NN for X3
fig = plt.figure(figsize=(5, 5))
plt.plot(tr_history3.epoch, tr_history3.history['loss'], 'b^--', linewidth=2, markersize=4, label=('Training'))
plt.plot(tr_history3.epoch, tr_history3.history['val_loss'], 'rv--', linewidth=2, markersize=4, label=('Validation'))
plt.xlabel('Epoch', fontsize = FS)
plt.ylabel(r'$\Sigma MSE$', fontsize = FS)
plt.ylim([0, 1.2*np.max(tr_history3.history['val_loss'])])
plt.title('Pressure Data', fontsize = FS)
plt.legend(loc='upper right', fontsize = FS)
fig.tight_layout()
# plt.savefig('G:\\My Drive\\RPI\\MANE 6962 Machine Learning\\Project\\Figures\\X3_loss.png', dpi=300)

fig = plt.figure(figsize=(5, 5))
plt.plot(tr_history3.epoch, tr_history3.history['mse'], 'b^--', linewidth=2, markersize=4, label=('Training'))
plt.plot(tr_history3.epoch, tr_history3.history['val_mse'], 'rv--', linewidth=2, markersize=4, label=('Validation'))
plt.xlabel('Epoch', fontsize = FS)
plt.ylabel('MSE', fontsize = FS)
plt.ylim([0, 1.2*np.max(tr_history3.history['val_mse'])])
plt.title('Pressure Data', fontsize = FS)
plt.legend(loc='upper right', fontsize = FS)
fig.tight_layout()
# plt.savefig('G:\\My Drive\\RPI\\MANE 6962 Machine Learning\\Project\\Figures\\X3_mse.png', dpi=300)

fig = plt.figure(figsize=(5, 5))
plt.plot(y_test[:,0], y_test[:,1], 'ko', label='Test Data')
plt.plot(y_pred3[:,0], y_pred3[:,1], 'rx', label='Pressure Data')
plt.ylabel(r'$C_l$', fontsize = FS)
plt.xlabel(r'$C_d$', fontsize = FS)
plt.xlim((-0.05, 0.35))
plt.ylim((-1.10, 1.4))
plt.legend(loc='center right', fontsize = FS)
fig.tight_layout()
# plt.savefig('G:\\My Drive\\RPI\\MANE 6962 Machine Learning\\Project\\Figures\\NNcompareX3.png', dpi=300)

fig = plt.figure(figsize=(5, 10))
plt.subplot(211)
plt.plot(y_test[:,0], y_test[:,1], 'ko', label='Test Data')
plt.plot(y_pred2[:,0], y_pred2[:,1], 'g*', label='Test Parameters')
plt.ylabel(r'$C_l$', fontsize = FS)
plt.title('Test Parameters', fontsize = FS)
plt.legend(loc='center right', fontsize = FS)
plt.subplot(212)
plt.plot(y_test[:,0], y_test[:,1], 'ko', label='Test Data')
plt.plot(y_pred3[:,0], y_pred3[:,1], 'rx', label='Pressure Data')
plt.ylabel(r'$C_l$', fontsize = FS)
plt.xlabel(r'$C_d$', fontsize = FS)
plt.title('Pressure Data', fontsize = FS)
plt.legend(loc='center right', fontsize = FS)
fig.tight_layout()
# plt.savefig('G:\\My Drive\\RPI\\MANE 6962 Machine Learning\\Project\\Figures\\NNcompareX2X3_1.png', dpi=300)

fig = plt.figure(figsize=(5, 5))
plt.plot(y_test[:,0], y_test[:,1], 'ko', label='Test Data')
plt.plot(y_pred2[:,0], y_pred2[:,1], 'g*', label='Test Parameters')
plt.plot(y_pred3[:,0], y_pred3[:,1], 'rx', label='Pressure Data')
plt.ylabel(r'$C_l$', fontsize = FS)
plt.xlabel(r'$C_d$', fontsize = FS)
plt.xlim((-0.05, 0.35))
plt.ylim((-1.10, 1.4))
plt.legend(loc='center right', fontsize = FS)
fig.tight_layout()
# plt.savefig('G:\\My Drive\\RPI\\MANE 6962 Machine Learning\\Project\\Figures\\NNcompareX2X3_2.png', dpi=300)

print(f'Duration: {time.time()-t1} seconds')