from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from hyperas.distributions import choice
from sklearn.metrics import mean_squared_error
from neuralnetwork import NeuralNetwork
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical
import numpy

class DenseLSTM(NeuralNetwork):
  """
    This is the class responsible for create keras LSTM. It can be Dense with several
    LSTM cells or a single LSTM cell without Dense Layer.
  """
  def __init__(self, look_back=12, dense=False, lstm_cells=1, input_shape=1):
    self.look_back = look_back
    self.dense = dense
    self.lstm_cells = lstm_cells
    self.input_shape = input_shape
    self.model = self.__create_model()

  def __create_model(self):
    model = Sequential()
    lstm_cells = 1 if not self.dense else self.lstm_cells
    model.add(LSTM(lstm_cells, input_shape=(self.input_shape, self.look_back)))
    if self.dense:
      model.add(Dense(1))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    print(model.summary())

    return model

  def __create_label(self, dataset, mean_of=0):
    dataX, dataY = [], []

    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    for i in range(len(dataset)-self.look_back):
        day_t = dataset[i:(i+self.look_back)]
        day_t1 = dataset[i + self.look_back]
        dataX.append(day_t)
        if day_t.mean(axis=mean_of)[mean_of] > day_t1[mean_of]:
            dataY.append(0)
        else:
            dataY.append(1)

    return numpy.array(dataX), numpy.array(dataY)

  def create_data_for_fit(self, dataset, train_proportion=0.66, mean_of=0):
    """
    Create the labels and reshape the data according to parameters of
    DenseLSTM(look_back, input_shape)

    Parameters
    ----------
    dataset : list
      List of one or several features [feature_1, feature_2, ....]
    train_proportion : float
      It should be between 0.0 and 1.0 and represent the proportion of
      data set to include in the test split.
    mean_of : int
      The dataset parameter can have several features. The label is
      generated based on the mean of one of this features along the
      interval decided by look_back parameter in DenseLSTM. So label[0] will be
      the value of mean(feature_1[0], ... , feature_1[look_back])
    """
    dataX, dataY = self.__create_label(dataset)

    train_size = int(len(dataset) * train_proportion)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size], dataset[train_size:len(dataset)]

    trainX, testX = dataX[0:train_size,:], dataX[train_size:len(dataX):]
    trainY, testY = dataY[0:train_size], dataY[train_size:len(dataX)]

    trainX = numpy.reshape(trainX, (trainX.shape[0], self.look_back, trainX.shape[2]))
    testX = numpy.reshape(testX, (testX.shape[0], self.look_back, testX.shape[2]))
    trainX = numpy.array([t.transpose() for t in trainX])
    testX = numpy.array([t.transpose() for t in testX])

    self.trainX = trainX; self.trainY = trainY
    self.testX = testX; self.testY = testY
