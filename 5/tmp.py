# Listing 5.1. Ilustracja działania sztucznego neuronu ze skokową funkcją aktywacyjną
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import rcParams
#from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.utils.vis_utils import plot_model

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 16
np.set_printoptions(precision=2)
x = np.arange(0, 1, 0.01)
y = x.copy()
X, Y = np.meshgrid(x, y)
wx = 0.1
wy = 0.3
S = wx * X + wy * Y
out = S > 0.15
fig, ax = plt.subplots(1, 1)
ax.imshow(out)
ticks = np.around(np.arange(-0.2, 1.1, 0.2), 3)
ax.set_xticklabels(ticks)
ax.set_yticklabels(ticks)
plt.gca().invert_yaxis()
plt.show()

# Listing 5.3. Załadowanie danych
from sklearn.datasets import load_iris

data = load_iris()
y = data.target
X = data.data
y = pd.Categorical(y)
y = pd.get_dummies(y).values
class_num = y.shape[1]

# Listing 5.4. Przykład tworzenia sekwencyjnego modelu sieci neuronowej
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

model = Sequential()
model.add(Dense(64, input_shape=(X.shape[1],), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(class_num, activation='softmax'))
learning_rate = 0.0001
model.compile(optimizer=Adam(learning_rate), loss='categorical_crossentropy', metrics=('accuracy'))
model.summary()
plot_model(model, to_file="my_model.png")

# Listing 5.5. Przykład uczenia sieci neuronowej
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model.fit(X_train, y_train, batch_size=32, epochs=500, validation_data=(X_test, y_test), verbose=2)

# Listing 5.6. Pobranie historii uczenia modelu oraz jej wizualizacja.
from matplotlib import pyplot as plt

historia = model.history.history
floss_train = historia['loss']
floss_test = historia['val_loss']
acc_train = historia['accuracy']
acc_test = historia['val_accuracy']
fig, ax = plt.subplots(1, 2, figsize=(20, 10))
epochs = np.arange(0, 500)
ax[0].plot(epochs, floss_train, label='floss_train')
ax[0].plot(epochs, floss_test, label='floss_test')
ax[0].set_title('Funkcje strat')
ax[0].legend()
ax[1].set_title('Dokladnosci')
ax[1].plot(epochs, acc_train, label='acc_train')
ax[1].plot(epochs, acc_test, label='acc_test')
ax[1].legend()

# Listing 5.7. Przykład zastosowania walidacji krzyżowej.
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
accs = []
scaler = StandardScaler()
for train_index, test_index in KFold(5).split(X_train):
    X_train_cv = X_train[train_index, :]
    X_test_cv = X_train[test_index, :]
    y_train_cv = y_train[train_index, :]
    y_test_cv = y_train[test_index, :]
    X_train_cv = scaler.fit_transform(X_train_cv)
    X_test_cv = scaler.transform(X_test_cv)
    model.fit(X_train_cv, y_train_cv, batch_size=32, epochs=100, validation_data=(X_test_cv, y_test_cv), verbose=2)
    y_pred = model.predict(X_test_cv).argmax(axis=1)
    y_test_cv = y_test_cv.argmax(axis=1)
    accs.append(accuracy_score(y_test_cv, y_pred))

# Listing 5.8. Ładowanie zbioru danych MNIST
from sklearn.datasets import load_digits

data = load_digits()
X = data.data
y = data.target



# Załadowanie danych
iris = load_iris()
X, y = iris.data, iris.target
y = pd.Categorical(y)
y = pd.get_dummies(y).values
class_num = y.shape[1]

# Definiowanie parametrów do przeszukiwania
param_grid = {
    'layers': [(64,), (64, 64), (64, 64, 64)],
    'activation': ['relu', 'tanh', 'sigmoid'],
    'optimizer': [Adam(learning_rate=0.001), Adam(learning_rate=0.0001)],
    'epochs': [100, 200, 300]
}

# Tworzenie modelu
def create_model(layers, activation, optimizer, epochs):
    model = Sequential()
    for l in layers:
        model.add(Dense(l, activation=activation, input_shape=(X.shape[1],)))
    model.add(Dense(class_num, activation='softmax'))
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Walidacja krzyżowa
kf = KFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(estimator=create_model,
                           param_grid=param_grid,
                           cv=kf,
                           scoring='accuracy',
                           verbose=2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

grid_search.fit(X_train, y_train)

print("Najlepsze parametry:", grid_search.best_params_)
print("Najlepsza dokładność:", grid_search.best_score_)

# Ewaluacja najlepszego modelu
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print("Dokładność na zbiorze testowym:", accuracy_score(y_test, y_pred.argmax(axis=1)))
