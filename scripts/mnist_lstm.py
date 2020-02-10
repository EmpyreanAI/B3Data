import sys
sys.path.append('../')

import matplotlib.pyplot as plt
import pandas
import os
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.datasets import mnist
import numpy
from tensorflow import keras

(train_x, train_y), (test_x, test_y) = mnist.load_data()

train_x = train_x / 255.0
test_x = test_x / 255.0

model = Sequential()
model.add(LSTM(100, input_shape=(train_x.shape[1:]), activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

model.fit(train_x, train_y, epochs=5, verbose=0, validation_split=0.33)
loss, acc = model.evaluate(test_x, test_y,
                             verbose=2)

# print('Accuracy: ' + str(acc))
