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


#zad1.3
plot_step = 0.01
# Define the range of x values and calculate y
x = np.arange(-5, 5+plot_step, plot_step)
y_1 = np.tanh(x)
y_2 = (np.exp(x)-np.exp(-x)) / (np.exp(x)+np.exp(-x))
y_3 = 1 / (1 + np.exp(-x))
y_4 = np.where(x <= 0, 0, x)
y_5 = np.where(x <= 0, np.exp(x)-1, x)

# Create the plot, add labels and title, show it
plt.plot(x, y_1, label='tanh(x)', color='blue')
plt.title('y = tanh(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(0, color='black', lw=0.5, ls='--')
plt.grid()
plt.legend()
plt.show()

fig, axs = plt.subplots(2, 2, figsize=(10, 8))
axs[0,0].plot(x, y_2, label='exp', color='red')
axs[0,0].set_title('y = (e^{x} - e^{-x}) / (e^{x} + e^{-x})')
axs[0,0].set_xlabel('x')
axs[0,0].set_ylabel('y')
axs[0,0].legend()
axs[0,0].axhline(0, color='black', lw=0.5, ls='--')
axs[0,0].axvline(0, color='black', lw=0.5, ls='--')
axs[0,0].grid()
axs[0,1].plot(x, y_3, label='exp', color='blue')
axs[0,1].set_title('y = 1 / (1 + e^{-x})')
axs[0,1].set_xlabel('x')
axs[0,1].set_ylabel('y')
axs[0,1].legend()
axs[0,1].axhline(0, color='black', lw=0.5, ls='--')
axs[0,1].axvline(0, color='black', lw=0.5, ls='--')
axs[0,1].grid()
axs[1,0].plot(x, y_4, label='where 1', color='black')
axs[1,0].set_title('y = {x; 0, x<=0')
axs[1,0].set_xlabel('x')
axs[1,0].set_ylabel('y')
axs[1,0].legend()
axs[1,0].axhline(0, color='blue', lw=0.5, ls='--')
axs[1,0].axvline(0, color='blue', lw=0.5, ls='--')
axs[1,0].grid()
axs[1,1].plot(x, y_5, label='where 2', color='green')
axs[1,1].set_title('y = {x; e^{x}-1, x <= 0')
axs[1,1].set_xlabel('x')
axs[1,1].set_ylabel('y')
axs[1,1].legend()
axs[1,1].axhline(0, color='black', lw=0.5, ls='--')
axs[1,1].axvline(0, color='black', lw=0.5, ls='--')
axs[1,1].grid()
plt.tight_layout()
plt.show()
