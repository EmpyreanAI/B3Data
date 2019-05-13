# LSTM for international airline passengers problem with regression framing
import numpy
import matplotlib.pyplot as plt
import pandas as pd
import math
from indicators import stochastic_oscilator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import glob

def prices(year=2015, cod='ELET3', month=1, period=6):
    path = "../BovespaWolf/data/COTAHIST_A" + str(year) + ".TXT.csv"
    files = glob.glob(path)

    for name in files:
        print(name)
        with open(name) as file:
            df = pd.read_csv(file, low_memory=False)

        df = df.drop(df.index[-1])

        df['DATA'] = df['DATA'].astype(str)
        df['TIPREG'] = df['TIPREG'].astype(int)
        df['CODBDI'] = df['CODBDI'].astype(int)
        df['CODNEG'] = df['CODNEG'].astype(str)


        df['TPMERC'] = df['TPMERC'].astype(int)

        df['NOMRES'] = df['NOMRES'].astype(str)
        df['ESPECI'] = df['ESPECI'].astype(str)
        df['PRAZOT'] = df['PRAZOT'].astype(str)
        df['MODREF'] = df['MODREF'].astype(str)


        df['PREABE'] = (df['PREABE']/100).astype(float)
        df['PREMAX'] = (df['PREMAX']/100).astype(float)
        df['PREMIN'] = (df['PREMIN']/100).astype(float)
        df['PREMED'] = (df['PREMED']/100).astype(float)
        df['PREULT'] = (df['PREULT']/100).astype(float)
        df['PREOFC'] = (df['PREOFC']/100).astype(float)
        df['PREOFV'] = (df['PREOFV']/100).astype(float)

        df['TOTNEG'] = df['TOTNEG'].astype(int)
        df['QUATOT'] = df['QUATOT'].astype(numpy.int64)
        df['VOLTOT'] = df['VOLTOT'].astype(numpy.int64)
        df['PREEXE'] = (df['PREEXE']/100).astype(float)
        df['INDOPC'] = df['INDOPC'].astype(int)
        df['DATAEN'] = df['DATAEN'].astype(str)
        df['FATCOT'] = df['FATCOT'].astype(int)
        df['PTOEXE'] = df['PTOEXE'].astype(numpy.int64)
        df['CODISI'] = df['CODISI'].astype(str)
        df['DISMES'] = df['DISMES'].astype(int)

        df['DATA'] = pd.to_datetime(df['DATA'],
                                        format='%Y%m%d',
                                        errors='ignore')

        filtered = df.loc[df['CODNEG'] == cod]

        next_month = (month + period) % 12

        stocks = filtered.loc[(filtered['DATA'] > str(year) + '-' + str(month) + '-01') & 
                              (filtered['DATA'] <= str(year) + '-' + str(next_month) + '-01')]
        
        return stocks['PREULT']

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

# fix random seed for reproducibility
numpy.random.seed(7)

# load the dataset
dataframe = prices()
dataset = dataframe.values
dataset = dataset.astype('float32')
dataset = numpy.asarray(stochastic_oscilator(dataset, 14))
dataset = dataset.reshape(len(dataset), 1)

# import pdb; pdb.set_trace()

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
import pdb; pdb.set_trace()

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()