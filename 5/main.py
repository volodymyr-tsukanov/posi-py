# -*- coding: utf-8 -*-
"""
Created on Wed Nov 6 15:16:24 2024

@author: volodymyr-tsukanov

needs:  pip install keras tensorflow pydot
"""
import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score

from keras.models import Sequential  # Import Sequential model for neural network
from keras.layers import Input, Dense  # Import layers for input and dense (fully connected) layers
from keras.optimizers import Adam, RMSprop, SGD  # Import optimizers
from keras.utils import plot_model  # Import utility for visualizing the model

from matplotlib import pyplot as plt


def visualize_model_history(model, model_epochs):
    """
    Visualizes the training history of a model by plotting loss and accuracy curves
    for both training and validation sets.

    Parameters:
    - model: Trained Keras model that has a history attribute containing training metrics.
    - model_epochs: The number of epochs used during training to set the range for the x-axis.
    
    Plots:
    - A figure with two subplots:
        1. Training and validation loss curves.
        2. Training and validation accuracy curves.
    """
    # Extract history
    historia = model.history.history
    floss_train = historia['loss']
    floss_test = historia['val_loss']
    acc_train = historia['accuracy']
    acc_test = historia['val_accuracy']
    
    # Create subplots
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    epochs = np.arange(0, model_epochs)
    
    # Plot loss
    ax[0].plot(epochs, floss_train, label='floss_train')
    ax[0].plot(epochs, floss_test, label='floss_test')
    ax[0].set_title('Loss Functions')
    ax[0].legend()

    # Plot accuracy
    ax[1].plot(epochs, acc_train, label='acc_train')
    ax[1].plot(epochs, acc_test, label='acc_test')
    ax[1].set_title('Accuracy')
    ax[1].legend()
    
    # Display the plots
    plt.show()



### Load data
data = load_iris()
y = data.target
X = data.data
y = pd.Categorical(y)     #one hot encoding
y = pd.get_dummies(y).values  #one hot encoding
class_num = y.shape[1]

model_epochs = 100
model_cross_validation_count = 3


### Init model
model = Sequential()
model.add(Dense(64, input_shape=(X.shape[1],), activation='relu'))  # Add a Dense layer with ReLU activation
model.add(Dense(64, activation='relu'))  # Add another Dense layer with ReLU activation
model.add(Dense(64, activation='relu'))  # Add another Dense layer with ReLU activation
model.add(Dense(class_num, activation='softmax'))  # Output layer with softmax for multi-class classification

learning_rate = 0.0001  # Set the learning rate for the optimizer
model.compile(optimizer=Adam(learning_rate),  # Compile model with Adam optimizer
              loss='categorical_crossentropy',  # Set loss function for multi-class classification
              metrics=['accuracy'])  # Monitor accuracy during training

model.summary()  # Display model summary (architecture details)
plot_model(model, to_file="data/my_model.png")  # Save a plot of the model architecture


### Set split + train model
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model.fit(X_train, y_train, batch_size=32, epochs=model_epochs,
validation_data=(X_test, y_test), verbose=2)


visualize_model_history(model, model_epochs)


### cross validation
accs = []
scaler = StandardScaler()
model_best = model

# K-Fold Cross Validation
for train_index, test_index in KFold(model_cross_validation_count).split(X_train):
    X_train_cv = X_train[train_index, :]
    X_test_cv = X_train[test_index, :]
    y_train_cv = y_train[train_index, :]
    y_test_cv = y_train[test_index, :]

    # Standardize the data
    X_train_cv = scaler.fit_transform(X_train_cv)
    X_test_cv = scaler.transform(X_test_cv)

    # Train the model
    model.fit(X_train_cv, y_train_cv, batch_size=32, epochs=model_epochs, validation_data=(X_test_cv, y_test_cv), verbose=2)

    # Make predictions and compute accuracy
    y_pred = model.predict(X_test_cv).argmax(axis=1)
    y_test_cv = y_test_cv.argmax(axis=1)
    acc_score = accuracy_score(y_test_cv, y_pred)
    accs.append(acc_score)
    if(acc_score > max(accs)):
        model_best = model
        print("!!! new best: ", acc_score)

print('Model accuracies: ', accs)
visualize_model_history(model_best, model_epochs)