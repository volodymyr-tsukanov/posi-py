#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 21:03:58 2024
"""
import os
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.svm import SVC as SVM
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score


def _qualitative_to_0_1(data, column, value_to_be_1):
    mask = data[column].values == value_to_be_1
    data[column][mask] = 1
    data[column][~mask] = 0
    return data

def _read_digits():
    return load_digits(return_X_y=True)

def _read_voice():
    data = pd.read_csv("voice_extracted_features.csv", sep=",")
    data = _qualitative_to_0_1(data, 'label', 'female')
    vals = data.values.astype(np.float64)
    return vals[:, :-1], vals[:, -1]

def get_data(name):
    if name == "digits":
        return _read_digits()
    elif name == "voice":
        return _read_voice()
    else:
        raise Exception("Wrong name!!!")

def train_ev_models(x_train, x_test, y_train, y_test):
    models = [kNN(), SVM(), DT(max_depth=4)]
    for model in models:
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(model)
        print(confusion_matrix(y_test, y_pred))
        print("ACC=", accuracy_score(y_test, y_pred))
        print("F1=", f1_score(y_test, y_pred))


if __name__ == "__main__":
    print(">>>> START")
    x, y = get_data("voice")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    print(">>>> ORG")
    train_ev_models(x_train, x_test, y_train, y_test)

    transformers = [PCA(0.95), PCA(7), FastICA(n_components=7)]
    labels = ["PCA 095 (10)", "PCA 7", "ICA 7"]

    for trans, label in zip(transformers, labels):
        x_train_tr = trans.fit_transform(x_train)
        x_test_tr = trans.transform(x_test)
        print(f"\n>>>> {label}\n")
        train_ev_models(x_train_tr, x_test_tr, y_train, y_test)

    print(">>>> END")


# Załaduj dane z pliku "voice_extracted_features.csv"
data = pd.read_csv("voice_extracted_features.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Podziel dane na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Oryginalne dane (ORG)
accuracy_org_knn = KNeighborsClassifier().fit(X_train, y_train).score(X_test, y_test)
accuracy_org_SVM = SVM().fit(X_train, y_train).score(X_test, y_test)
accuracy_org_dtc = DT().fit(X_train, y_train).score(X_test, y_test)

print(f"Accuracy ORG: kNN={accuracy_org_knn}, SVM={accuracy_org_SVM}, DTC={accuracy_org_dtc}")

# PCA 095
pca_095 = PCA(0.95)
X_train_pca_095 = pca_095.fit_transform(X_train)
X_test_pca_095 = pca_095.transform(X_test)
accuracy_pca_095_knn = KNeighborsClassifier().fit(X_train_pca_095, y_train).score(X_test_pca_095, y_test)
accuracy_pca_095_SVM = SVM().fit(X_train_pca_095, y_train).score(X_test_pca_095, y_test)
accuracy_pca_095_dtc = DT().fit(X_train_pca_095, y_train).score(X_test_pca_095, y_test)

print(f"Accuracy PCA 095: kNN={accuracy_pca_095_knn}, SVM={accuracy_pca_095_SVM}, DTC={accuracy_pca_095_dtc}")

# PCA 7
pca_7 = PCA(n_components=7)
X_train_pca_7 = pca_7.fit_transform(X_train)
X_test_pca_7 = pca_7.transform(X_test)
accuracy_pca_7_knn = KNeighborsClassifier().fit(X_train_pca_7, y_train).score(X_test_pca_7, y_test)
accuracy_pca_7_SVM = SVM().fit(X_train_pca_7, y_train).score(X_test_pca_7, y_test)
accuracy_pca_7_dtc = DT().fit(X_train_pca_7, y_train).score(X_test_pca_7, y_test)

print(f"Accuracy PCA 7: kNN={accuracy_pca_7_knn}, SVM={accuracy_pca_7_SVM}, DTC={accuracy_pca_7_dtc}")

# ICA 7
ica_7 = FastICA(n_components=7, random_state=0)
X_train_ica_7 = ica_7.fit_transform(X_train)
X_test_ica_7 = ica_7.transform(X_test)
accuracy_ica_7_knn = KNeighborsClassifier().fit(X_train_ica_7, y_train).score(X_test_ica_7, y_test)
accuracy_ica_7_SVM = SVM().fit(X_train_ica_7, y_train).score(X_test_ica_7, y_test)
accuracy_ica_7_dtc = DT().fit(X_train_ica_7, y_train).score(X_test_ica_7, y_test)

print(f"Accuracy ICA 7: kNN={accuracy_ica_7_knn}, SVM={accuracy_ica_7_SVM}, DTC={accuracy_ica_7_dtc}")



# Listing 4.5. Wygenerowanie wykresu procentu wyjaśnionej wariancji oraz znalezienie liczby
# składowych głównych, zapewniającej 95 wyjaśnionej wariancji.
def qualitative_to_0_1(data,column,value_to_be_1):
    mask=data[column].values==value_to_be_1
    data[column][mask]=1
    data[column][~mask]=0
    return data

data=pd.read_csv("voice_extracted_features.csv",sep=',')
data=qualitative_to_0_1(data,'label','female')
features=list(data.columns)
vals=data.values.astype(np.float64)
X=vals[:,:-1]
y=vals[:,-1]
X_train,X_test, y_train, y_test=train_test_split(X,y,test_size=0.2, shuffle=False)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
pca_transform = PCA()
pca_transform.fit(X_train)
variances = pca_transform.explained_variance_ratio_
cumulated_variances = variances.cumsum()
plt.scatter(np.arange(variances.shape[0]), cumulated_variances)
plt.yticks(np.arange(0, 1.1, 0.1))
PC_num = (cumulated_variances<0.95).sum()
print(PC_num)
plt.show()

# Listing 4.6. Przykład zastosowania analizy składowych głównych oraz wizualizacji jej wyników
X_paced=PCA(2).fit_transform(X_train)
fig,ax=plt.subplots(1,1)
females=y_train==1
ax.scatter(X_paced[females,0],X_paced[females,1], label='female')
ax.scatter(X_paced[~females,0],X_paced[~females,1], label='male')
ax.legend()
plt.show()

# Listing 4.7. Przykład zastosowania metody ICA.
from sklearn.datasets import load_digits
from sklearn.decomposition import FastICA
X, _ = load_digits(return_X_y=True)
transformer = FastICA(n_components=7, random_state=0)
X_transformed = transformer.fit_transform(X)

# Listing 4.8. Przykład klasyfikacji z zastosowaniem PCA oraz skalowania danych
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier as kNN
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6, stratify=y)
pca_transform = PCA(9)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train_pcaed = pca_transform.fit_transform(X_train)
X_test_pcaed = pca_transform.transform(X_test)
X_train_pcaed_scaled = scaler.fit_transform(X_train_pcaed)
X_test_pcaed_scaled = scaler.transform(X_test_pcaed)
model = kNN(5, weights = 'distance')
model.fit(X_train_pcaed_scaled, y_train)
y_predict = model.predict(X_test_pcaed_scaled)

# Listing 4.9. Przykład zastosowania obiektu klasy Pipeline
from sklearn.pipeline import Pipeline
pipe = Pipeline([['transformer', PCA(9)], ['scaler', StandardScaler()], ['classifier', kNN(weights='distance')]])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)



# Zadanie 4.1
# Wczytanie danych
data = pd.read_csv('voice_extracted_features.csv')

# Przygotowanie danych
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Analiza PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Wizualizacja PCA
plt.figure(figsize=(10, 6))
plt.scatter(X_train_pca[y_train == 0, 0], X_train_pca[y_train == 0, 1], label='Mężczyzna')
plt.scatter(X_train_pca[y_train == 1, 0], X_train_pca[y_train == 1, 1], label='Kobieta')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.title('Pierwsze dwie składowe główne')
plt.show()

# Ustalenie optymalnej liczby składowych głównych
pca = PCA()
pca.fit(X_train)
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Liczba składowych głównych')
plt.ylabel('Odsetek wyjaśnionej wariancji')
plt.title('Wykres procentu wyjaśnionej wariancji')
plt.axhline(y=0.95, color='r', linestyle='--')
plt.show()
n_components = np.sum(np.cumsum(pca.explained_variance_ratio_) < 0.95) + 1
print(f'Optymalna liczba składowych głównych: {n_components}')

# Zbudowanie modelu klasyfikacji z użyciem Pipeline
pipe = Pipeline([
    ('pca', PCA(n_components=n_components)),
    ('scaler', StandardScaler()),
    ('classifier', KNeighborsClassifier(n_neighbors=5, weights='distance'))
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# Zadanie 4.2
num_experiments = 30
knn_accuracies = []
svm_accuracies = []
dt_accuracies = []

for _ in range(num_experiments):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # KNN
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    knn.fit(X_train, y_train)
    knn_accuracies.append(knn.score(X_test, y_test))

    # SVM
    from sklearn.svm import SVM

    svm = SVM()
    svm.fit(X_train, y_train)
    svm_accuracies.append(svm.score(X_test, y_test))

    # Decision Tree
    from sklearn.tree import DT

    dt = DT()
    dt.fit(X_train, y_train)
    dt_accuracies.append(dt.score(X_test, y_test))

print('Średnie dokładności:')
print(f'KNN: {np.mean(knn_accuracies)}')
print(f'SVM: {np.mean(svm_accuracies)}')
print(f'Decision Tree: {np.mean(dt_accuracies)}')


# Zadanie 4.3
class OptimalPCAComponents(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.95):
        self.threshold = threshold
        self.pca = None
        self.n_components_ = None

    def fit(self, X, y=None):
        self.pca = PCA()
        self.pca.fit(X)
        self.n_components_ = np.sum(np.cumsum(self.pca.explained_variance_ratio_) < self.threshold) + 1
        return self

    def transform(self, X):
        return self.pca.transform(X)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


# Zadanie 4.4
class OutlierRemover(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_cleaned = X.copy()
        for col in X.T:
            mean = np.mean(col)
            std = np.std(col)
            X_cleaned = X_cleaned[np.abs(col - mean) < 3 * std]
        return X_cleaned