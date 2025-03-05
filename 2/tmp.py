#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 20:33:12 2024
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.datasets import load_diabetes


# Listing 2.2. Przykład wczytania, podziału na wejście/wyjście oraz podziału na treningowy i testowy podzbiór.
bh_data = pd.read_csv('practice_lab_2.csv', sep=';')
bh_arr = bh_data.values
X, y = bh_arr[:,:-1], bh_arr[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=221, shuffle=False)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Listing 2.3. Przykład tworzenia oraz uczenia modelu regresji liniowej
linReg = LinearRegression()
linReg.fit(X_train, y_train)
y_pred = linReg.predict(X_test)
minval = min(y_test.min(), y_pred.min())
maxval = max(y_test.max(), y_pred.max())
plt.scatter(y_test, y_pred)
plt.plot([minval, maxval], [minval, maxval])
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.show()

# Listing 2.4. Przykład zastosowania funkcji metryk regresji
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=221, shuffle=True)
linReg = LinearRegression()
linReg.fit(X_train, y_train)
y_pred = linReg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error (y_test, y_pred)
print(f"MSE: {mse}, MAE: {mae}, MAPE: {mape}")

# Listing 2.5. Przykład generacji wykresu pudełkowego dla cechy zależnej
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, shuffle=True)
plt.boxplot(y_train)
plt.title("Medianowa wartosc mieszkania")
plt.show()

# Listing 2.6. Przykład usunięcia oraz zastąpienia wartości odstających cechy zależnej
outliers = np.abs((y_train - y_train.mean())/
y_train.std())>3
X_train_no_outliers = X_train[~outliers,:]
y_train_no_outliers = y_train[~outliers]
y_train_mean = y_train.copy()
y_train_mean[outliers] = y_train.mean()

# Listing 2.7. Przykład generacji wykresu słupkowego wag poszczególnych cech niezależnych.
# linReg = LinearRegression()
# linReg.fit(X_train, y_train_mean)
# niezależne_cechy = bh_cechy[:-1]
# fig, ax = plt.subplots(1,1)
# x = np.arange(len(niezleżne_cechy))
# wagi = linReg.coef_
# ax.bar(x, wagi)
# ax.set_xticks(x)
# ax.set_xticklabels(niezleżne_cechy, rotation = 90)
# plt.show()


# Wczytanie danych
df = pd.read_csv('practice_lab_2.csv', sep=';')

# Generowanie macierzy korelacji
correlation_matrix = df.corr()

# Wizualizacja macierzy korelacji
plt.figure(figsize=(12, 10))
plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
plt.colorbar()
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.title('Macierz korelacji dla zbioru Housing')
for i in range(len(correlation_matrix.columns)):
    for j in range(len(correlation_matrix.columns)):
        plt.text(j, i, f"{correlation_matrix.iloc[i, j]:.2f}", ha='center', va='center')
plt.tight_layout()
plt.show()

# Wykresy korelacji między cechami niezależnymi a ceną
independent_features = df.columns.drop('MedianowaCena')
fig, axes = plt.subplots(4, 4, figsize=(20, 20))
axes = axes.flatten()

for i, feature in enumerate(independent_features):
    axes[i].scatter(df[feature], df['MedianowaCena'])
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('MedianowaCena')
    axes[i].set_title(f'{feature} vs MedianowaCena')

plt.tight_layout()
plt.show()


def test_linear_regression(n_repetitions):
    df = pd.read_csv('practice_lab_2.csv', sep=';')
    X = df.drop('MedianowaCena', axis=1).values
    y = df['MedianowaCena'].values

    mape_scores = []

    for _ in range(n_repetitions):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        mape_scores.append(mape)

    return np.mean(mape_scores)


# Przykładowe użycie
average_mape = test_linear_regression(100)
print(f"Średni procentowy błąd bezwzględny (MAPE): {average_mape:.2%}")


# Generacja nowych cech
# Spróbuj zaproponować cechy/kombinacje cech, które mogłyby ulepszyć jakość predykcji regresji liniowej.
def generate_new_features(X):
    new_features = np.column_stack([
        X[:, 4] / X[:, 7],  # TlenkiAzotu / LPokojow
        X[:, 4] / X[:, 5],  # TlenkiAzotu / WiekMieszkania
        X[:, 4] * X[:, 3],  # TlenkiAzotu * PrzyRzece
        X[:, 4] / X[:, -1]  # TlenkiAzotu / PracFiz
    ])
    return np.column_stack([X, new_features])


def test_linear_regression(n_repetitions, handle_outliers='none', use_new_features=False):
    df = pd.read_csv('practice_lab_2.csv', sep=';')
    X = df.drop('MedianowaCena', axis=1).values
    y = df['MedianowaCena'].values

    if use_new_features:
        X = generate_new_features(X)

    mape_scores = []

    for _ in range(n_repetitions):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        if handle_outliers != 'none':
            z_scores = np.abs((y_train - np.mean(y_train)) / np.std(y_train))
            outliers = z_scores > 3

            if handle_outliers == 'remove':
                X_train = X_train[~outliers]
                y_train = y_train[~outliers]
            elif handle_outliers == 'replace':
                y_train[outliers] = np.mean(y_train)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        mape_scores.append(mape)

    return np.mean(mape_scores)


# Przykładowe użycie
average_mape_original = test_linear_regression(100, 'none', False)
average_mape_new_features = test_linear_regression(100, 'none', True)
average_mape_new_features_remove_outliers = test_linear_regression(100, 'remove', True)

print(f"MAPE z oryginalnymi cechami: {average_mape_original:.2%}")
print(f"MAPE z nowymi cechami: {average_mape_new_features:.2%}")
print(f"MAPE z nowymi cechami i usuwaniem wartości odstających: {average_mape_new_features_remove_outliers:.2%}")


# Załadowanie danych
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
feature_names = diabetes.feature_names

# Tworzenie DataFrame
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# Analiza korelacji
correlation_matrix = df.corr()

# Wizualizacja macierzy korelacji
plt.figure(figsize=(12, 10))
plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
plt.colorbar()
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.title('Macierz korelacji dla zbioru Diabetes')
for i in range(len(correlation_matrix.columns)):
    for j in range(len(correlation_matrix.columns)):
        plt.text(j, i, f"{correlation_matrix.iloc[i, j]:.2f}", ha='center', va='center')
plt.tight_layout()
plt.show()

# Wykresy korelacji między cechami a zmienną docelową
fig, axes = plt.subplots(3, 4, figsize=(20, 15))
axes = axes.flatten()

for i, feature in enumerate(feature_names):
    axes[i].scatter(df[feature], df['target'])
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('target')
    axes[i].set_title(f'{feature} vs target')

plt.tight_layout()
plt.show()


# Funkcja do testowania modelu regresji liniowej
def test_linear_regression(n_repetitions, handle_outliers='none'):
    mape_scores = []

    for _ in range(n_repetitions):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        if handle_outliers != 'none':
            z_scores = np.abs((y_train - np.mean(y_train)) / np.std(y_train))
            outliers = z_scores > 3

            if handle_outliers == 'remove':
                X_train = X_train[~outliers]
                y_train = y_train[~outliers]
            elif handle_outliers == 'replace':
                y_train[outliers] = np.mean(y_train)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        mape_scores.append(mape)

    return np.mean(mape_scores)


# Testowanie modelu
average_mape = test_linear_regression(100)
average_mape_remove_outliers = test_linear_regression(100, 'remove')
average_mape_replace_outliers = test_linear_regression(100, 'replace')
