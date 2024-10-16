# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 18:36:46 2024

@author: volodymyr-tsukanov
"""
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


### Load cvs
data_csv = pd.read_csv("../2/practice_lab_2.csv", sep=";")
d = data_csv.corr()


### Array operations
arr = data_csv.values
rowsEven = arr[1::2,:]
rowsOdd = arr[0::2,:]
colZeroCount = (arr == 0).sum(axis=0)
colMaxIndexes = np.where(arr == arr.max())[1]   #row => [0]


### Draw corr matrix
def draw_corr(corr_data : pd.DataFrame, title='Corr-Mtx'):
    cmap = LinearSegmentedColormap.from_list("WiRGn", [(0.85,0.75,0.85,1), (1,0,0,1), (0,1,0,1)])
    
    plt.figure(figsize=(corr_data.shape[0], corr_data.shape[1]))
    plt.title(title, fontsize=22)
    plt.imshow(corr_data, cmap=cmap, aspect='equal')
    plt.colorbar()  #right bar
    plt.xticks(range(len(corr_data.columns)), corr_data.columns, rotation=45)   #axis labels
    plt.yticks(range(len(corr_data.columns)), corr_data.columns)
    for (i, j), val in np.ndenumerate(corr_data):   #data labels
        plt.text(j, i, f'{val:.2f}', ha='center', va='center', color=(0.05,0.15,0,1))
    plt.grid(False)  #disable gridlines
    plt.show()

