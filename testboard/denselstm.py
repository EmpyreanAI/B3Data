"""Nani."""

import numpy
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

from neuralnetwork import NeuralNetwork


class DenseLSTM(NeuralNetwork):
    """Class responsible for create keras LSTM.

    It can be Dense with several LSTM cells or a single LSTM cell
    without Dense Layer.
    """

    def __init__(self, look_back=12, dense=False, lstm_cells=1, input_shape=1):
        """Nani."""
        self.look_back = look_back
        self.dense = dense
        self.lstm_cells = lstm_cells
        self.input_shape = input_shape
        super(DenseLSTM, self).__init__()

    def _create_model(self):
        """Nani."""
        model = Sequential()
        lstm_cells = 1 if not self.dense else self.lstm_cells
        model.add(LSTM(lstm_cells, input_shape=(self.input_shape,
                                                self.look_back)))
        if self.dense:
            model.add(Dense(1))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['acc'])

        return model

    def _create_label(self, dataset, mean_of=0):
        """Nani."""
        data_x, data_y = [], []

        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)

        for i in range(len(dataset)-self.look_back):
            day_t = dataset[i:(i+self.look_back)]
            day_t1 = dataset[i + self.look_back]
            data_x.append(day_t)
            if day_t.mean(axis=mean_of)[mean_of] > day_t1[mean_of]:
                data_y.append(0)
            else:
                data_y.append(1)

        return numpy.array(data_x), numpy.array(data_y)

    def set_look_back(self, value):
        """Nani."""
        self.look_back = value

    def create_data_for_fit(self, dataset, train_proportion=0.66):
        """Create the labels and reshape data for fit.

        Create the labels and reshape data according to parameters of
        DenseLSTM(look_back, input_shape).

        Parameters
        ----------
        dataset : list
          List of one or several features [feature_1, feature_2, ....]
        train_proportion : float
          It should be between 0.0 and 1.0 and represent the proportion of
          data set to include in the test split.

        """
        data_x, data_y = self._create_label(dataset)
        train_size = int(len(dataset) * train_proportion)

        train_x = data_x[0:train_size]
        test_x = data_x[train_size:len(data_x)]
        train_y = data_y[0:train_size]
        test_y = data_y[train_size:len(data_x)]

        train_x = numpy.reshape(train_x, (train_x.shape[0], self.look_back,
                                          train_x.shape[2]))
        test_x = numpy.reshape(test_x, (test_x.shape[0], self.look_back,
                                        test_x.shape[2]))
        train_x = numpy.array([t.transpose() for t in train_x])
        test_x = numpy.array([t.transpose() for t in test_x])

        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
