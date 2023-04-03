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
# Remove Flap column
print(df[df.columns[:14]])
print(df[df.columns[15:]])
df = pd.concat([df[df.columns[:14]], df[df.columns[15:]]], axis=1)

# Save to .csv file
out_flnm1 = 'C:\\Users\\John Wylie\\Documents\\RPI\\Courses\\MANE 6962 Machine Learning\\Project Git\\wind-turbine-regression\\df_data.csv'
df.to_csv(out_flnm1, sep='\t')

out_flnm2 = 'C:\\Users\\John Wylie\\Documents\\RPI\\Courses\\MANE 6962 Machine Learning\\Project Git\\wind-turbine-regression\\np_data.csv'
df.to_csv(out_flnm2, sep='\t')
## Convert df to numpy array and pre-process features