# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 14:24:35 2024

@author: volodymyr-tsukanov
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#zad1.2
data_csv = pd.read_csv("practice_lab_1.csv", sep=";")

arr_colNames = data_csv.columns.to_numpy()
arr1 = data_csv.values
arr1_colEven = arr1[1::2,:]   #parzyste
arr1_colOdd = arr1[0::2,:]   #nieparzyste
arr2 = arr1_colEven - arr1_colOdd    #róźnica

arr2_2 = (arr2 - arr2.mean()) / arr2.std()

print('p2 ', arr2_2[:3,:3])

arr2_3 = (arr2 - arr2.mean(axis=0)) / np.spacing(arr2.std(axis=0))

print('p3 ', arr2_3[:3,:3])

arr2_4 = arr1.mean(axis=0) / np.spacing(arr1.std(axis=0))

print('p4 ', arr2_4)

print('p5 ', arr2_4.max())

arr2_6 = (arr1 > arr2_4).sum(axis=0)    #axis=0 -> along column

print('p6 ', arr2_6)

arr1_colMaxIndexes = np.where(arr1 == arr1.max())[1]
arr2_7 = arr_colNames[np.unique(arr1_colMaxIndexes)]

print('p7 ', arr2_7)

arr1_colZeroCount = (arr1 == 0).sum(axis=0)
arr2_8 = arr_colNames[arr1_colZeroCount > 0]

print('p8 ', arr2_8)

arr2_9 = arr_colNames[arr1_colEven.sum(axis=0) > arr1_colOdd.sum(axis=0)]

print('p9 ', arr2_9)
