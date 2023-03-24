# John Wylie, RIN# 661262436

import os
from os.path import exists
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




# flnm = 'G:\\My Drive\\RPI\\MANE 6962 Machine Learning\\Project\\Data\\01.Wind-Tunnel\\Aerodynamic-Data\\corr\\01_corr_N63418-Re3clean.txt'
# flnm = 'G:\\My Drive\\RPI\\MANE 6962 Machine Learning\\Project\\Data\\01.Wind-Tunnel\\Aerodynamic-Data\\corr\\14_corr_N63-418-p3wrap_P40_4%.txt'
# if exists(flnm):
#     print('success')
#     df = pd.read_csv(flnm, sep='\t', header=0, skiprows = [1])
#     # df = pd.read_csv(flnm, sep='\t', index_col=False, header=0, skiprows=1, on_bad_lines='skip')
# print(df)


filedir = 'G:\\My Drive\\RPI\\MANE 6962 Machine Learning\\Project\\Data\\01.Wind-Tunnel\\Aerodynamic-Data\\corr'
files = os.listdir(filedir)

df_list = {}
for i in range(len(files)):
    flnm = filedir + '\\' + files[i]
    if exists(flnm):
        print('Success')
        df_list[i] = pd.read_csv(flnm, sep='\t', header=0, skiprows = [1])
        print(flnm)
