# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 14:20:49 2024

@author: volodymyr-tsukanov


housing.csv => practice_lab_2.csv
random_state = 221
shuffle = True
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


data_csv = pd.read_csv("practice_lab_2.csv", sep=";")

#zad2.1
cr_mx = data_csv.corr()

f = plt.figure(figsize=(14, 14))
plt.matshow(cr_mx, fignum=f.number)
plt.imshow(cr_mx, cmap='coolwarm', interpolation='nearest')
plt.xticks(range(data_csv.select_dtypes(['number']).shape[1]), data_csv.select_dtypes(['number']).columns, fontsize=14, rotation=45)
plt.yticks(range(data_csv.select_dtypes(['number']).shape[1]), data_csv.select_dtypes(['number']).columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=10)
plt.title('Correlation Matrix', fontsize=16);
plt.show()


#zad2.2
bh_cechy = data_csv.columns.to_list()
bh_arr = data_csv.values
X, y = bh_arr[:,:-1], bh_arr[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=221, shuffle=True)
linReg = LinearRegression()
linReg.fit(X_train, y_train)
y_pred = linReg.predict(X_test)

minval = min(y_test.min(), y_pred.min())
maxval = max(y_test.max(), y_pred.max())
plt.scatter(y_test, y_pred)
plt.plot([minval, maxval], [minval, maxval])
plt.xlabel('y_test')
plt.ylabel('y_pred')

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error (y_test, y_pred)
