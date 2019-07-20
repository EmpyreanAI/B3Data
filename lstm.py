import numpy
from stocks import Stocks
from label_helper import LabelHelper
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import GRU
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

train_proportion = 0.66

look_back = 12  

numpy.random.seed(7)

dataframe = Stocks.prices(cod='PETR3')
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

model = Sequential()
model.add(LSTM(1, input_shape=(1, look_back)))
model.compile(loss='mean_squared_error',
              optimizer='adam', 
              metrics=['accuracy'])
print(model.summary())
model.fit(trainX, trainY, epochs=5000, batch_size=1, verbose=2, callbacks = callbacks)

scores = model.evaluate(testX, testY, verbose=2)
print("Accuracy: %.2f%%" % (scores[1]*100))


