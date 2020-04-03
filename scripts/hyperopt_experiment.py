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
from hyperopt import hp, fmin, tpe, hp, STATUS_OK, Trials
import numpy
from hyperopt.plotting import main_plot_vars
from hyperopt import base
from keras import backend as K

space = {
    'batch_size': hp.choice('batch_size', [1, 2, 32, 64, 128, 256]),
    'cells': hp.choice('cells', [1, 50, 80, 100, 150, 200]),
    'optimizers': hp.choice('optimizers', ['sgd','adam','rmsprop']),
    'look_back_proportion': hp.choice('look_back_proportion', [1, 3, 6, 9, 12]),
    'nb_epochs' :  5000,
}

stocks = Stocks(year=2014, cod=sys.argv[1], period=5)
dataset = stocks.selected_fields([CLOSING])

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

    train_proportion = 0.7
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
    model.add(LSTM(params['cells'], input_shape=(1, look_back)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=params['optimizers'],
                  metrics=['acc',f1_m, precision_m, recall_m])


    model.fit(train_x, train_y, batch_size=params['batch_size'], epochs=params['nb_epochs'],
                   verbose=0, validation_split=0.33)
    loss, acc, f1_score, precision, recall = model.evaluate(test_x, test_y,
                                 verbose=2)

    K.clear_session()

    return {'loss': -f1_score, 'status': STATUS_OK }

trials = Trials()
best = fmin(objective, space, algo=tpe.suggest, trials=trials, max_evals=1000)
df = pandas.DataFrame()
trial_dict = {}
for t in trials.trials:
    trial_dict.update(t['misc']['vals'])
    trial_dict.update(t['result'])
    df = df.append(trial_dict, ignore_index=True)

print (best)
print (trials.best_trial)

outname = 'hyperopt_100_' + sys.argv[1] + '.csv'
outdir = '../results'
if not os.path.exists(outdir):
    os.mkdir(outdir)

fullname = os.path.join(outdir, outname)

df.to_csv(fullname, mode='a')

best_df = pandas.DataFrame()
best_df = best_df.append(trials.best_trial, ignore_index=True)

outname = 'best_trial_' + sys.argv[1] + '.csv'
if not os.path.exists(outdir):
    os.mkdir(outdir)

fullname = os.path.join(outdir, outname)

best_df.to_csv(fullname, mode='a')
