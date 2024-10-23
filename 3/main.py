# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 14:16:36 2024

@author: volodymyr-tsukanov


X_train, Y_train <= from **lab2**
random_state=221, shuffle = False
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.svm import SVC as SVM
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

def qualitative_to_0_1(data, column, value_to_be_1):
    mask = data[column].values == value_to_be_1
    data.loc[mask, column] = 1
    data.loc[~mask, column] = 0
    return data


data_csv = pd.read_csv("practice_lab_3.csv", sep=";")
data = qualitative_to_0_1(data_csv, 'Gender', 'Female')
data = qualitative_to_0_1(data,'Married','Yes')
data = qualitative_to_0_1(data,'Education','Graduate')
data = qualitative_to_0_1(data,'Self_Employed','Yes')
data = qualitative_to_0_1(data,'Loan_Status','Y')

cat_feature = pd.Categorical(data.Property_Area)
one_hot = pd.get_dummies(cat_feature)
data = pd.concat([data, one_hot], axis = 1)
data = data.drop(columns = ['Property_Area'])

features = data.columns
y = data['Loan_Status'].values.astype(np.float64)
x = data.drop(columns = ['Loan_Status']).values.astype(np.float64)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=221, shuffle=False)

models = [kNN(), SVM(), DT(max_depth=4)]
for model in models:
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    print(model)
    print(confusion_matrix(y_test,y_pred))
    print(accuracy_score(y_test,y_pred))
    print(f1_score(y_test,y_pred))

plt.figure(figsize=(20,10))
tree_vis = plot_tree(models[2],feature_names=data_csv.columns,class_names=['N','Y'],fontsize=12)
#tree_vis = plot_tree(model, feature_names=data.columns[:-1].to_list(), class_names=['N','Y'], fontsize = 20)
