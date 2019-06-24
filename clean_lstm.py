import numpy
import math
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

train_proportion = 0.66

numpy.random.seed(7)

dataframe = pd.read_csv('./k-fold-lstm/price_label.csv')

prices = numpy.reshape(dataframe.price.values, (-1, 1))
prices = dataframe.price.values
# import pdb; pdb.set_trace()
labels = dataframe.label.values

# prices_mean = numpy.mean(prices)
# deviation = numpy.std(prices)

# prices = [((prices[i]-prices_mean)/deviation) for i in range(len(prices))]

# scaler = MinMaxScaler(feature_range=(0, 1))
# prices = scaler.fit_transform(prices)

prices = numpy.reshape(prices, (-1, 1))
scaler = StandardScaler(copy=True)
scaler.fit(prices)
prices = scaler.transform(prices)


# import pdb; pdb.set_trace()
train_size = int(len(dataframe) * train_proportion) 
test_size = len(dataframe) - train_size
train, test = dataframe[0:train_size], dataframe[train_size:len(dataframe)]

trainX, testX = prices[0:train_size,:], prices[train_size:len(prices):]
trainY, testY = labels[0:train_size], labels[train_size:len(prices)]

trainX = numpy.reshape(trainX, (trainX.shape[0], 1, 1))
testX = numpy.reshape(testX, (testX.shape[0], 1, 1))

model = Sequential()
model.add(LSTM(1, input_shape=(1, 1)))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

scores = model.evaluate(testX, testY, verbose=2)
print("Accuracy: %.2f%%" % (scores[1]*100))


