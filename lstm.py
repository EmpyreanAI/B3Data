# LSTM for international airline passengers problem with regression framing
import numpy
import matplotlib.pyplot as plt
import math
import indicators
import pandas as pd
from stocks import Stocks
from helpers import Helpers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import GRU
from keras.utils import np_utils
from keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

df = pd.DataFrame()

# fix random seed for reproducibility
numpy.random.seed(7)

# import pdb; pdb.set_trace()
# list_indicators = dir(indicators)[9:]
list_indicators = []
list_indicators.append("price")

for indicator in list_indicators:
        # load the dataset
        dataframe = Stocks.prices()
        dataset = Helpers.regenerate_dataset(dataframe, indicator)

        # normalize the dataset
        import pdb; pdb.set_trace()
        scaler = MinMaxScaler(feature_range=(0, 1))
        full_dataset = scaler.fit_transform(dataset)

        k = 10
        fold_jump = len(dataset)//k

        # for i in range(1, k+1):
        # dataset = full_dataset[:fold_jump*i]

        # split into train and test sets
        train_size = int(len(dataset) * 1)
        test_size = len(dataset) - train_size
        train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

        # reshape into X=t and Y=t+1
        look_back = 1
        trainX, trainY = Helpers.create_dataset(train, look_back)
        testX, testY = Helpers.create_dataset(test, look_back)
        import pdb; pdb.set_trace()
        print(trainY)
        print(testY)

        # reshape input to be [samples, time steps, features]
        trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        # import pdb; pdb.set_trace()

        # create and fit the LSTM network
        model = Sequential()
        model.add(LSTM(1, input_shape=(1, look_back)))
        # model.add(Dense(1, activation='sigmoid', input_shape=(1, look_back)))
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

        scores = model.evaluate(testX, testY, verbose=2)
        print("Accuracy: %.2f%%" % (scores[1]*100))

        # register = {'k': i, 'windows_size': windows_size, 'accuracy': "%.2f%%" % (scores[1]*100)}

        # df = df.append(register, ignore_index=True)

        # make predictions
        # trainPredict = model.predict(trainX)
        # testPredict = model.predict(testX)

        # print("Trains predict: ")
        # print(trainPredict)
        # print("Test predict: ")
        # print(testPredict)


# invert predictions

# trainPredict = scaler.inverse_transform(trainPredict)
# trainY = to_zero_one(trainY)
# testPredict = scaler.inverse_transform(testPredict)
# testY = to_zero_one(testY)


# calculate root mean squared error

# trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
# print('Train Score: %.2f RMSE' % (trainScore))
# testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
# print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting

# trainPredictPlot = numpy.empty_like(dataset)
# trainPredictPlot[:, :] = numpy.nan
# trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# # shift test predictions for plotting

# testPredictPlot = numpy.empty_like(dataset)
# testPredictPlot[:, :] = numpy.nan
# testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

# # plot baseline and predictions

# plt.plot(scaler.inverse_transform(dataset))
# plt.plot(trainPredictPlot)
# plt.plot(testPredictPlot)
# # plot.scatter(b[1], b[0], label='skitscat', color='green', s=25, marker="^")
# plt.show()
