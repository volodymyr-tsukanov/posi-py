#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 20:22:50 2024
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data_csv = pd.read_csv("practice_lab_1.csv", sep=";")


# Listing 1.4. Przykład operacji na tablicach numpy.
arr = np.array([[2, 3, 5, 1],
                [5, 1, 2, 8],
                [5, 1, 6, -1]])
pojedynczy_element = arr[0, 3]
pojedynczy_element_od_konca = arr[-3, -1]

# Listing 1.5. Przykłady indeksowania zaawansowanego.
kolumna = arr[:, 0]
wiersz = arr[2, :]

obszar = arr[0:2, 1:4]
obszar_pomijajac_koncowe_indeksy = arr[:2, 1:]
parzyste_kolumny = arr[:, ::2]
nieparzyste_kolumny = arr[:, 1::2]
odwrocone_kolumny = arr[:, ::-1]

# Listing 1.6. Przykład pobrania wybranych kolumn za pomocą tablicy indeksów
indeksy = [1, 1, -1]
wybrane_kolumny = arr[:, indeksy]
maska = arr > 2  # Zwróci tablicę bool
# np.False_	np.True_	np.True_	np.False_
# np.True_	np.False_	np.False_	np.True_
# np.True_	np.False_	np.True_	np.False_

arr_maskowane = arr[maska]

# Listing 1.7. Przykłady operacji wektorowych na tablicach numpy.
arr1 = np.array([1, 2, 3, 4, 5, 6])
arr2 = np.array([7, 8, 9, 10, 11, 12])
suma = arr1 + arr2  # wynik: [8, 10, 12, 14, 16, 18]
roznica = arr1 - arr2  # wynik: [-6, -6, -6, -6, -6, -6]
arr1_razy_2 = 2 * arr1  # wynik: [2, 4, 6, 8, 10, 12]
arr1_plus_2 = 2 + arr1  # wynik: [3, 4, 5, 6, 7, 8]
arr1_do_kwadratu = arr1 ** 2  # wynik: [1,4,9,16,25,36]
suma_kwadratow = arr1 ** 2 + arr2 ** 2  # wynik: [50,68,90,116,146,180]

# Listing 1.8. Przykłady zastosowania metod klasy ndarray.
suma_arr1 = arr1.sum()  # wynik: 21
suma_sum = (arr1 + arr2).sum()  # wynik: 78
iloczyn_sum = (arr1 + arr2).prod()  # wynik: 3870720

# Listing 1.9. Przykłady zastosowania metod klasy ndarray na tablicach dwuwymiarowych.
arr = np.array([[2, 3, 5, 1],
                [5, 1, 2, 8],
                [5, 1, 6, -1]])
sumy_kolumn = arr.sum(axis=0)  # wynik: [12, 5, 13, 8]
sumy_wierszy = arr.sum(axis=1)  # wynik: [11, 16, 11]

# Listing 1.10. Przykład implementacji bardziej zaawansowanego wzoru za pomocą metod tablicy numpy
arr1_0_1 = (arr1 - arr1.min()) / (arr1.max() - arr1.min())
print(arr1)  # wyświetli [1 2 3 4 5 6]
print(arr1_0_1)  # wyświetli [0. 0.2 0.4 0.6 0.8 1. ]

# Listing 1.11. Przykład przeskalowania wartości w tablicy dwuwymiarowej.
arr_0_1 = (arr - arr.min(axis=0)) / (arr.max(axis=0) - arr.min(axis=0))
print(arr_0_1)
# Wyświetli:
#  [[0. 1. 0.75 0.22222222]
#  [1. 0. 0. 1. ]
#  [1. 0. 1. 0. ]]

# Listing 1.14. Przykład generacji wykresów za pomocą modułu pyplot biblioteki matplotlib.
x = np.arange(0, 10, 0.1)
y = np.sin(x ** 2 - 5 * x + 3)
plt.scatter(x, y)
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')

# Listing 1.15. Przykład wykonania kilku wykresów na jednym obrazie.
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(x, y)
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")
ax[1].scatter(x, y)
ax[1].set_xlabel("x")
ax[1].set_ylabel("y")
fig.tight_layout()
plt.show()

# Listing 1.16. Przykład utworzenia wielu wykresów w postaci dwuwymiarowej tablicy
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
ax[0, 0].scatter(x, y)
ax[0, 1].plot(x, y)
ax[1, 0].hist(y)
ax[1, 1].boxplot(y)


data = data_csv.values
column_names = list(data_csv.columns)
# print(data_csv.head())
# print(data_csv.columns)
# print(list(data_csv.columns))
# print(data_csv.values)
# print(type(data_csv.values))

# Różnica między parzystymi a nieparzystymi wierszami:
even_rows = data[::2]
odd_rows = data[1::2]
result_1 = even_rows - odd_rows
# print("Różnica między parzystymi a nieparzystymi wierszami: " + str(result_1))

# Przekształcanie danych
result_2 = (data - np.mean(data)) / np.std(data)
# print("Przekształcenie danych: " + str(result_2))

# Przekształcanie danych dla oddzielnych kolumn
result_3 = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + np.spacing(np.std(data, axis=0)))
# print("Przekształcenie danych dla oddzielnych kolumn: " + str(result_3))

# Obliczenie współczynnika zmienności dla każdej kolumny:
mean = np.mean(data, axis=0)
std = np.std(data, axis=0)
result_4 = mean / (std + np.spacing(std))
print("Współczynniki zmienności dla każdej kolumny: " + str(result_4))

# Znalezienie kolumny o największym współczynniku zmienności:
max_cv_column = np.argmax(result_4)
print(f"Kolumna o największym współczynniku zmienności: {column_names[max_cv_column]}")

# Liczba elementów większych od średniej dla każdej kolumny:
result_6 = np.sum(data > np.mean(data, axis=0), axis=0)
print("Liczba elementów większych od średniej dla każdej kolumny: " + str(result_6))

# Nazwy kolumn z wartością maksymalną:
max_value = np.max(data)
max_columns = np.array(column_names)[np.any(data == max_value, axis=0)]
print(f"Kolumny z wartością maksymalną: {max_columns}")

# Nazwy kolumn z największą liczbą zer:
zero_counts = np.sum(data == 0, axis=0)
max_zero_columns = np.array(column_names)[zero_counts == np.max(zero_counts)]
print(f"Kolumny z największą liczbą zer: {max_zero_columns}")

# Nazwy kolumn, gdzie suma elementów parzystych jest większa od sumy nieparzystych:
even_sum = np.sum(data[::2], axis=0)
odd_sum = np.sum(data[1::2], axis=0)
result_9 = np.array(column_names)[even_sum > odd_sum]
print(f"Kolumny, gdzie suma parzystych > suma nieparzystych: {result_9}")


# a) f(x) = tanh(x)
x = np.arange(-5, 5, 0.01)
y = np.tanh(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title("f(x) = tanh(x)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid()
plt.show()

# b) f(x) = (e^x-e^(-x)) / (e^x + e^(-x))
x = np.arange(-5, 5, 0.01)
y = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title("f(x) = (e^x-e^(-x)) / (e^x + e^(-x))")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid()
plt.show()

# c) f(x) = 1 / (1 + e^(-x))
x = np.arange(-5, 5, 0.01)
y = 1 / (1 + np.exp(-x))

plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title("f(x) = 1 / (1 + e^(-x))")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid()
plt.show()

# d) f(x) = { x, x > 0
#           { 0, x <= 0
x = np.arange(-5, 5, 0.01)
y = np.where(x > 0, x, 0)

plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title("f(x) = { x, x > 0 \n           { 0, x <= 0")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid()
plt.show()

# e) f(x) = { x, x > 0
#           { e^x - 1, x <= 0
x = np.arange(-5, 5, 0.01)
y = np.where(x > 0, x, np.exp(x) - 1)

plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title("f(x) = { x, x > 0 \n           { e^x - 1, x <= 0")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid()
plt.show()
