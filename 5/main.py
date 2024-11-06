# -*- coding: utf-8 -*-
"""
Created on Wed Nov 6 15:16:24 2024

@author: volodymyr-tsukanov
"""
import pandas as pd

from sklearn.datasets import load_iris

from keras.models import Sequential  # Import Sequential model for neural network
from keras.layers import Input, Dense  # Import layers for input and dense (fully connected) layers
from keras.optimizers import Adam, RMSprop, SGD  # Import optimizers
from keras.utils import plot_model  # Import utility for visualizing the model



### Load csv
data = load_iris()
y = data.target
X = data.data
y = pd.Categorical(y)     #one hot encoding
y = pd.get_dummies(y).values  #one hot encoding
class_num = y.shape[1]


model = Sequential()  # Initialize a Sequential model
model.add(Dense(64, input_shape=(X.shape[1],), activation='relu'))  # Add a Dense layer with ReLU activation
model.add(Dense(64, activation='relu'))  # Add another Dense layer with ReLU activation
model.add(Dense(64, activation='relu'))  # Add another Dense layer with ReLU activation
model.add(Dense(class_num, activation='softmax'))  # Output layer with softmax for multi-class classification

learning_rate = 0.0001  # Set the learning rate for the optimizer
model.compile(optimizer=Adam(learning_rate),  # Compile model with Adam optimizer
              loss='categorical_crossentropy',  # Set loss function for multi-class classification
              metrics=('accuracy'))  # Monitor accuracy during training

model.summary()  # Display model summary (architecture details)
plot_model(model, to_file="my_model.png")  # Save a plot of the model architecture
