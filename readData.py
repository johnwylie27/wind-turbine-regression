# John Wylie, RIN# 661262436
# This script reads in the relevant project data and adds roughness-related
# features before normalizing, shifting, and pre-processing the data.

import os
from os.path import exists
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Check File Directory
filedir = 'G:\\My Drive\\RPI\\MANE 6962 Machine Learning\\Project\\Data\\01.Wind-Tunnel\\Aerodynamic-Data\\corr'
files = os.listdir(filedir)

## Load Data from Files into List of Data Frames
df_list = {}
for i in range(len(files)):
    flnm = filedir + '\\' + files[i]
    if exists(flnm):
        print('Success')
        df_list[i] = pd.read_csv(flnm, sep='\t', header=0, skiprows = [1])
        print(flnm)

## Load Test List from Excel
testlist = pd.read_excel('G:\\My Drive\\RPI\\MANE 6962 Machine Learning\\Project\\Data\\01.Wind-Tunnel\\Aerodynamic-Data\\Runlog.xlsx')

## Add features from testlist to dataframes

## Combine data frames into one single data frame

## Convert df to numpy array and pre-process features