import sys
sys.path.append('../')

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from testboard.stocks import Stocks
from testboard.stocks import CLOSING, OPENING, MAX_PRICE, MIN_PRICE, MEAN_PRICE, VOLUME
from hyperopt import hp, fmin, tpe, hp, STATUS_OK, Trials
import numpy

space = {
    'recurrent_dropout': hp.uniform('recurrent_dropout', 0, 1),
    'batch_size': hp.choice('batch_size', [1, 2, 64, 128, 256, 512]),
    'cells': hp.choice('cells', [1, 2, 16, 20, 50, 80, 100]),
    'look_back_proportion': hp.choice('look_back_proportion', [25, 50, 75, 100]),
    'nb_epochs' :  10000,
}

stocks = Stocks(year=2015, cod='PETR3', period=11)
dataset = stocks.selected_fields([CLOSING])

def label(dataset, look_back_proportion, mean_of=0):
    """Nani."""
    data_x, data_y = [], []

    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    look_back = int(len(dataset)*0.3*(look_back_proportion/100))

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

def objective(params):
    train_x, train_y, test_x, test_y, look_back = create_data_set(look_back_proportion=params['look_back_proportion'])

    model = Sequential()
    model.add(LSTM(1, input_shape=(1, look_back),
                      recurrent_dropout=params['recurrent_dropout']))
    model.add(Dense(1))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])


    model.fit(train_x, train_y, batch_size=params['batch_size'], epochs=params['nb_epochs'],
                   verbose=0, validation_split=0.33)
    loss, acc = model.evaluate(test_x, test_y,
                                 verbose=2)

    return {'loss': -loss, 'status': STATUS_OK }


trials = Trials()

best = fmin(objective, space, algo=tpe.suggest, trials=trials, max_evals=1)

print (best)
print (trials.best_trial)
