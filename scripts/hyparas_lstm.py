import numpy
from stocks import Stocks, CLOSING
from hyperas import optim
from label_helper import LabelHelper
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from hyperopt import Trials, STATUS_OK, tpe
from hyperas.distributions import choice
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils

def data():
  look_back = 12
  train_proportion = 0.66
  stocks = Stocks()
  dataframe = stocks.selected_fields([CLOSING])
  prices, labels = LabelHelper.up_down_with_interval(dataframe.values, look_back=look_back)

  scaler = MinMaxScaler(feature_range=(0, 1))
  prices = scaler.fit_transform(prices)

  train_size = int(len(dataframe) * train_proportion)
  test_size = len(dataframe) - train_size
  train, test = dataframe[0:train_size], dataframe[train_size:len(dataframe)]

  trainX, testX = prices[0:train_size,:], prices[train_size:len(prices):]
  trainY, testY = labels[0:train_size], labels[train_size:len(prices)]

  trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
  testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

  return trainX, trainY, testX, testY

def create_model(trainX, trainY, testX, testY):
  look_back = 12
  model = Sequential()
  model.add(LSTM(1, input_shape=(1, look_back)))
  model.compile(loss='mean_squared_error',
                optimizer={{choice(['rmsprop', 'adam', 'sgd'])}},
                metrics=['accuracy'])
  print(model.summary())
  model.fit(trainX, trainY, epochs=5000, batch_size={{choice([1, 16, 32, 64])}}, verbose=0)
  score, acc = model.evaluate(testX, testY, verbose=0)
  print('Test accuracy:', acc)
  return {'loss': -acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
  train_proportion = 0.66
  look_back = 12
  numpy.random.seed(7)

  best_run, best_model = optim.minimize(model=create_model,
                                        data=data,
                                        algo=tpe.suggest,
                                        max_evals=5,
                                        trials=Trials())

  trainX, trainY, testX, testY = data()
  print("Evalutation of best performing model:")
  print(best_model.evaluate(testX, testY))
  print("Best performing model chosen hyper-parameters:")
  print(best_run)
