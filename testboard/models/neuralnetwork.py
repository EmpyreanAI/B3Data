"""Nani."""

import numpy
from hyperopt import STATUS_OK
from sklearn.preprocessing import MinMaxScaler
from models.helpers.callbacks import LossHistory


class NeuralNetwork():
    """Nani."""

    def __init__(self):
        """Nani."""
        self.train_x = []
        self.train_y = []
        self.test_x = []
        self.test_y = []
        self.model = self._create_model()

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

    def fit_and_evaluate(self, epochs):
        """Nani."""
        history_train = LossHistory()
        self.model.fit(self.train_x, self.train_y, batch_size=256,
                       epochs=epochs, verbose=1, callbacks=[history_train])

        loss, acc = self.model.evaluate(self.test_x, self.test_y,
                                        batch_size=256, verbose=0)
        self.log('Test Loss:' + str(loss))
        self.log('Test Accuracy:' + str(acc))
        return {'acc': acc, 'loss': history_train.losses,
                'status': STATUS_OK, 'model': self.model}

    @staticmethod
    def log(message):
        """Nani."""
        print('[NeuralNetwork] ' + message)
