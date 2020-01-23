import sys
sys.path.append('../')

import matplotlib.pyplot as plt
import pandas
import os
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from testboard.data_mining.stocks import Stocks
from testboard.data_mining.stocks import CLOSING, OPENING, MAX_PRICE, MIN_PRICE, MEAN_PRICE, VOLUME
from keras import backend as K
import numpy

stocks = Stocks(year=2014, cod=sys.argv[1], period=5)
dataset = stocks.selected_fields([CLOSING])

batch_size = int(sys.argv[2])
cells = int(sys.argv[3])
look_back = int(sys.argv[4])
optimizer = sys.argv[5]

def label(dataset, look_back_proportion, mean_of=0):
    """Nani."""
    data_x, data_y = [], []

    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    look_back = look_back_proportion

    for i in range(len(dataset)-look_back):
        day_t = dataset[i:(i+look_back)]
        day_t1 = dataset[i + look_back]
        data_x.append(day_t)
        if day_t.mean(axis=mean_of)[mean_of] > day_t1[mean_of]:
            data_y.append(0)
        else:
            data_y.append(1)

    return numpy.array(data_x), numpy.array(data_y), look_back

def create_data_set(look_back_proportion):

    train_proportion = 0.33
    data_x, data_y, look_back = label(dataset, look_back_proportion)
    train_size = int(len(dataset) * train_proportion)

    train_x = data_x[0:train_size]
    test_x = data_x[train_size:len(data_x)]
    train_y = data_y[0:train_size]
    test_y = data_y[train_size:len(data_x)]

    train_x = numpy.reshape(train_x, (train_x.shape[0], train_x.shape[1],
                                      train_x.shape[2]))
    test_x = numpy.reshape(test_x, (test_x.shape[0], test_x.shape[1],
                                    test_x.shape[2]))
    train_x = numpy.array([t.transpose() for t in train_x])
    test_x = numpy.array([t.transpose() for t in test_x])

    return train_x, train_y, test_x, test_y, look_back

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

model = Sequential()
model.add(LSTM(cells, input_shape=(1, look_back)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['acc', f1_m, precision_m, recall_m])

train_x, train_y, test_x, test_y, look_back = create_data_set(look_back_proportion=look_back)

model.fit(train_x, train_y, batch_size=batch_size, epochs=5000,
               verbose=2, validation_split=0.33)
loss, acc, f1_score, precision, recall = model.evaluate(test_x, test_y,
                             verbose=2)

print("F1_Score: " + str(f1_score))
print("Acc: " + str(acc))
