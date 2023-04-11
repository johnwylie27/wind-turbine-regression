# John Wylie, RIN# 661262436
# This script reads in the relevant project data and adds roughness-related
# features before normalizing, shifting, and pre-processing the data.

import os
from os.path import exists
# import math
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

# def dfRmNaN(df):
#     for i in range(len(df.columns)): # loop through the columns
#         if 1:
#             print()
#     return df


## Check File Directory
filedir = 'C:\\Users\\John Wylie\\Documents\\RPI\\Courses\\MANE 6962 Machine Learning\\Project Git\\wind-turbine-regression\\Aerodynamic-Data\\corr'
files = os.listdir(filedir)

## Load Test List from Excel
fileLog = 'C:\\Users\\John Wylie\\Documents\\RPI\\Courses\\MANE 6962 Machine Learning\\Project Git\\wind-turbine-regression\\Aerodynamic-Data\\Runlog.csv'
testlist = pd.read_csv(fileLog)
testlist.fillna(0, inplace=True)
## Load Data from Files into List of Data Frames
df_list = {}
ind = 0
for i in range(len(files)):
    flnm = filedir + '\\' + files[i]
    if exists(flnm):
        print('Success')
        temp = pd.read_csv(flnm, sep='\t', header=0, skiprows = [1], index_col=False)
        temp.fillna(0, inplace=True)
        m = len(temp)
        
        ## Add new features to temporary dataframe
        temp['Grade'] = testlist.iloc[i, 1]*np.ones(m)
        temp['Grit Size'] = testlist.iloc[i, 2]*np.ones(m)
        temp['Lower Coverage'] = testlist.iloc[i, 3]*np.ones(m)
        temp['Upper Coverage'] = testlist.iloc[i, 4]*np.ones(m)
        
        ## Adjust indexing
        if i != 0:
            temp.index = temp.index + ind
            ind = ind + len(temp)
        else:
            ind = len(temp)
        df_list[i] = temp
        
        ## Add dataframe to master dataframe
        if i == 0:
            df = df_list[i]
        else:
            frames = [df, df_list[i]]
            df = pd.concat(frames)
        print(flnm)
        
## Remove Cn, Ct, Cd-press, Cl-press, Flap, Cn-corr, Ct-corr, Cm-corr columns
# Column numbers to skip: 6-9, 14, 77-79

# Input variables: Alpha, M, Q, V, Z_w.r., Cp values, Grade, Grit Size, Lower/Upper Cov.
out_temp1 = pd.concat([df[df.columns[1]], df[df.columns[10:14]], df[df.columns[15:78]], df[df.columns[81:]]], axis=1)  
# Output variables: C_l, C_d, C_m
out_temp2 = pd.concat([df[df.columns[2:5]]])

## Save to .csv file
out_flnm = ['C:\\Users\\John Wylie\\Documents\\RPI\\Courses\\MANE 6962 Machine Learning\\Project Git\\wind-turbine-regression\\df_data_X.csv']
out_flnm.append('C:\\Users\\John Wylie\\Documents\\RPI\\Courses\\MANE 6962 Machine Learning\\Project Git\\wind-turbine-regression\\df_data_y.csv')
out_temp1.to_csv(out_flnm[0], sep=',')
out_temp2.to_csv(out_flnm[1], sep=',')