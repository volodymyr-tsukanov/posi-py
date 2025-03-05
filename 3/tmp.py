#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 20:56:24 2024
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.svm import SVC as SVM
from sklearn.tree import DecisionTreeClassifier as DT, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score


data = pd.read_csv('practice_lab_3.csv', sep=';')

# Zadanie 3.2 - Funkcja do konwersji cech binarnych
def qualitative_to_0_1(data, column, value_to_be_1):
    mask = data[column].values == value_to_be_1
    data.loc[mask, column] = 1
    data.loc[~mask, column] = 0
    return data

# Przekształcenie wszystkich binarnych cech jakościowych
binary_features = {
    'Gender': 'Female',
    'Married': 'Yes',
    'Education': 'Graduate',
    'Self_Employed': 'Yes',
    'Credit_History': 'Yes',
    'Loan_Status': 'Y'
}

for column, value in binary_features.items():
    data = qualitative_to_0_1(data, column, value)

# Przekształcenie Property_Area na one-hot encoding
cat_feature = pd.Categorical(data.Property_Area)
one_hot = pd.get_dummies(cat_feature)
data = pd.concat([data, one_hot], axis=1)
data = data.drop(columns=['Property_Area'])

# Splitting data
y = data['Loan_Status'].values.astype(np.float64)
x = data.drop(columns=['Loan_Status']).values.astype(np.float64)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=221, shuffle=False)


# Zadanie 3.3 - Classification metrics calculation
def calculate_metrics(confusion_matrix):
    """
    Oblicza metryki klasyfikacji na podstawie macierzy pomyłek

    Args:
        confusion_matrix: macierz pomyłek w formacie numpy array

    Returns:
        dict: słownik zawierający wartości metryk
    """
    TP = confusion_matrix[1, 1]
    TN = confusion_matrix[0, 0]
    FP = confusion_matrix[0, 1]
    FN = confusion_matrix[1, 0]

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    specificity = TN / (FP + TN) if (FP + TN) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

    return {
        'Sensitivity': sensitivity,
        'Precision': precision,
        'Specificity': specificity,
        'Accuracy': accuracy,
        'F1': f1
    }

# Zadanie 3.4 - Decision Tree visualization
def plot_decision_tree(model, x, y):
    plt.figure(figsize=(20, 10))
    plot_tree(model, filled=True, feature_names=data.columns, class_names=['No', 'Yes'])
    plt.show()

models = [kNN(), SVM(), DT(max_depth=4)]

for model in models:
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(model)
    print(confusion_matrix(y_test, y_pred))
    # print(accuracy_score(y_test, y_pred))
    # print(f1_score(y_test, y_pred))
    print(calculate_metrics(confusion_matrix(y_test, y_pred)))
    print("\n")

plot_decision_tree(models[2], x_train, y_train)

def test_scaling_methods():
    print("\nScaling Methods Comparison:")
    for scaler in [StandardScaler(), MinMaxScaler(), RobustScaler()]:
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        for model in [kNN(), SVM()]:
            model.fit(x_train_scaled, y_train)
            y_pred = model.predict(x_test_scaled)
            print(f"\n{model.__class__.__name__} with {scaler.__class__.__name__}:")
            print(calculate_metrics(confusion_matrix(y_test, y_pred)))

test_scaling_methods()