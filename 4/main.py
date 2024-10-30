# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 18:36:46 2024

@author: volodymyr-tsukanov
"""
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA, FastICA
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.pipeline import Pipeline


##### ---CLASSES---
class PCAOptimalComponents:    # wyznacza optymalną liczbę składowych głównych dla PCA
    def __init__(self, variance_threshold=0.95):
        self.variance_threshold = variance_threshold
        self.pca = PCA()
        self.n_components_ = None

    def fit(self, X):
        # Dopasowanie PCA do danych
        self.pca.fit(X)
        # Obliczenie skumulowanej wariancji
        cumulative_variance = np.cumsum(self.pca.explained_variance_ratio_)
        # Wyznaczenie optymalnej liczby komponentów
        self.n_components_ = np.argmax(cumulative_variance >= self.variance_threshold) + 1
        return self

    def transform(self, X):
        # Transformacja danych przy użyciu optymalnej liczby komponentów
        return self.pca.transform(X)[:, :self.n_components_]

    def fit_transform(self, X):
        # Dopasowanie i transformacja w jednym kroku
        self.fit(X)
        return self.transform(X)


##### ---FUNCTIONS---
def draw_corr(corr_data : pd.DataFrame, title='Corr-Mtx'):
    """
    Draws correlation matrix of a DataFrame.
    
    Parameters
    ----------
    corr_data : pandas.DataFrame
    title : TYPE, optional
    - The default is 'Corr-Mtx'.
    """
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

def qualitative_to_0_1(data : pd.DataFrame, column : str, value_to_be_1 : str):
    """
    Non numeric(qualitative ; bool or status) feature to num (0 or 1)

    Parameters
    ----------
    data : pandas.DataFrame
    column : str (String)
    value_to_be_1 : str (String)

    Returns
    -------
    new pandas.DataFrame with modified column values

    """
    mask = data[column].values == value_to_be_1
    data.loc[mask, column] = 1
    data.loc[~mask, column] = 0
    return data


##### ---MAIN---
### Load cvs
data_csv = pd.read_csv("voice_extracted_features.csv", sep=",")

data = qualitative_to_0_1(data_csv, 'label', 'female')
features = list(data.columns)
vals = data.values.astype(np.float)
# Separate features (X) and target variable (y)
X = vals[:, :-1]
y = vals[:, -1]
# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Scale the features to have zero mean and unit variance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Apply PCA to reduce dimensionality of the feature set
pca_transform = PCA()
pca_transform.fit(X_train)
variances = pca_transform.explained_variance_ratio_
cumulated_variances = variances.cumsum()

# Draw a plot for cumulated_variances
plt.scatter(np.arange(variances.shape[0]), cumulated_variances)
plt.yticks(np.arange(0, 1.1, 0.1))
PC_num = (cumulated_variances < 0.95).sum() + 1    #+1 to cover 0.95...0.95999 values
print(PC_num)

# Apply PCA to reduce the dimensionality of the training data to 2 dimensions
X_paced = PCA(2).fit_transform(X_train)

fig, ax = plt.subplots(1, 1)
females = y_train == 1
ax.scatter(X_paced[females, 0], X_paced[females, 1], label='female')
ax.scatter(X_paced[~females, 0], X_paced[~females, 1], label='male')
ax.legend()

X, _ = load_digits(return_X_y=True)
transformer = FastICA(n_components=7, random_state=0)
X_transformed = transformer.fit_transform(X)


## ! new train & test !
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6, stratify=y)

pca_transform = PCA(9)
scaler = StandardScaler()

# Fit the scaler to the training data and transform it
X_train = scaler.fit_transform(X_train)
# Transform the test data using the fitted scaler
X_test = scaler.transform(X_test)
# Fit PCA on the training data and transform it to 9 principal components
X_train_pcaed = pca_transform.fit_transform(X_train)
# Transform the test data using the fitted PCA
X_test_pcaed = pca_transform.transform(X_test)
# Scale the PCA-transformed training data
X_train_pcaed_scaled = scaler.fit_transform(X_train_pcaed)
# Scale the PCA-transformed test data using the same scaler
X_test_pcaed_scaled = scaler.transform(X_test_pcaed)

# Initialize the kNN classifier with 5 neighbors, using distance-based weighting
model = kNN(5, weights='distance')
model.fit(X_train_pcaed_scaled, y_train)
y_predict = model.predict(X_test_pcaed_scaled)


## (Zad4.3)
pipe =Pipeline([['transformer', PCA(9)], ['scaler', StandardScaler()], ['pca_optimal', PCAOptimalComponents(variance_threshold=0.95)] ['classifier', kNN(weights='distance')]])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

print("Optymalna liczba składowych głównych:", pipe.named_steps['pca_optimal'].n_components_)
