import sys
sys.path.append('../')

import matplotlib.pyplot as plt
import pandas
import os
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from numpy import array

# %% markdown
# A **Markdown** cell!

(train_x, train_y), (test_x, test_y) = mnist.load_data()

train_x = train_x / 255.0
test_x = test_x / 255.0

# model = Sequential()
# model.add(LSTM(100, input_shape=(train_x.shape[1:]), activation='relu', return_sequences=True, return_state=True))
# model.add(Dense(10, activation='softmax'))
# model.compile(loss='sparse_categorical_crossentropy',
#               optimizer='adam',
#               metrics=['acc'])
# model.summary()



inputs1 = Input(shape=(3, 1))
lstm1, state_h, state_c = LSTM(100, return_sequences=True, return_state=True)(inputs1)
model = Model(inputs=inputs1, outputs=[lstm1, state_h, state_c])
# define input data
data = array([0.1, 0.2, 0.3]).reshape((1,3,1))
# make and show prediction
# print(model.predict(data))
model.summary()
# model.fit(train_x, train_y, epochs=3, verbose=2, validation_split=0.33)
# loss, acc = model.evaluate(test_x, test_y,
                             # verbose=2)

# print('Accuracy: ' + str(acc))
