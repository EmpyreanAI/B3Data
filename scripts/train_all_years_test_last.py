# Treinar com todos os anos e testar em 2019
# Treinar um ano e testar no próximo
# Treinar modelos por trimestre/bimestre/semestre por ano
# Repetir os testes com label de proximo dia e de media

import sys
sys.path.append('../')

import os
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense


import numpy
import matplotlib.pyplot as plt
from testboard.data_mining.stocks import Stocks
from testboard.data_mining.stocks import CLOSING
from testboard.validators.sequecialkfold import SequencialKFold
from keras import backend as K
import pandas
from tqdm import tqdm

def label_mean(dataset, look_back, mean_of=0):
    """Nani."""
    data_x, data_y = [], []

    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    for i in range(len(dataset)-look_back):
        day_t = dataset[i:(i+look_back)]
        day_t1 = dataset[i + look_back]
        data_x.append(day_t)
        if day_t.mean(axis=mean_of)[mean_of] > day_t1[mean_of]:
            data_y.append(0)
        else:
            data_y.append(1)

    return numpy.array(data_x), numpy.array(data_y), look_back

def label_next_day(dataset, look_back, mean_of=0):
    """Nani."""
    data_x, data_y = [], []

    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    for i in range(len(dataset)-look_back):
        day_t = dataset[i:(i+look_back)]
        day_t1 = dataset[i + look_back]
        data_x.append(day_t)
        if day_t[-1][mean_of] > day_t1[mean_of]:
            data_y.append(0)
        else:
            data_y.append(1)

    return numpy.array(data_x), numpy.array(data_y), look_back

def create_data_set(look_back, mean_label):

    stocks = Stocks(year=2018, cod=model['code'], period=11)
    dataset_test = stocks.selected_fields([CLOSING])

    if mean_label is 'mean':
        data_x, data_y, look_back = label_mean(dataset, look_back)
        data_test_x, data_test_y, look_back = label_mean(dataset_test, look_back)
    else:
        data_x, data_y, look_back = label_next_day(dataset, look_back)
        data_test_x, data_test_y, look_back = label_next_day(dataset_test, look_back)

    train_x = data_x
    train_y = data_y
    test_x = data_test_x
    test_y = data_test_y

    train_x = numpy.reshape(train_x, (train_x.shape[0], train_x.shape[1],
                                      train_x.shape[2]))
    test_x = numpy.reshape(test_x, (test_x.shape[0], test_x.shape[1],
                                    test_x.shape[2]))
    train_x = numpy.array([t.transpose() for t in train_x])
    test_x = numpy.array([t.transpose() for t in test_x])

    return train_x, train_y, test_x, test_y, look_back

models = [
    {'code': 'PETR3', 'window_size': 9, 'batch_size': 2, 'lstm_units': 80, 'optimizer': 'rmsprop'},
    {'code': 'VALE3', 'window_size': 6, 'batch_size': 2, 'lstm_units': 50, 'optimizer': 'rmsprop'},
    {'code': 'ABEV3', 'window_size': 6, 'batch_size': 128, 'lstm_units': 1, 'optimizer': 'adam'}
]

label_choice = ['mean', 'next_day']

df = pandas.DataFrame(columns=['PETR3', 'ABEV3', 'VALE3'], index=label_choice)
for value in label_choice:
    for model in tqdm(models):
        years = numpy.arange(2014, 2018)


        dataset = []
        for year in years:
            stocks = Stocks(year=year, cod=model['code'], period=11)
            dataset_aux = stocks.selected_fields([CLOSING])
            dataset.extend(dataset_aux)

        train_x, train_y, test_x, test_y, look_back = create_data_set(model['window_size'], value)

        lstm = Sequential()
        lstm.add(LSTM(model['lstm_units'], input_shape=(1, look_back)))
        lstm.add(Dense(1, activation='sigmoid'))
        lstm.compile(loss='binary_crossentropy',
                      optimizer=model['optimizer'],
                      metrics=['acc'])


        lstm.fit(train_x, train_y, batch_size=model['batch_size'], epochs=5000,
                  verbose=0, validation_split=0.33)
        loss, acc = lstm.evaluate(test_x, test_y,
                                   verbose=2)

        df.loc[value, [model['code']]] = acc

        print(df)
        K.clear_session()

df.to_csv('../results/train_all_test_last' + ".csv")
