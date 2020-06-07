"""Nani."""

import numpy
from hyperopt import STATUS_OK
from sklearn.preprocessing import MinMaxScaler


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

    def create_data_for_fit(self, dataset, train_proportion=0.7):
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

    # def evaluate_predictions(self, labels, predictions):
    #     """Nani."""
    #     confusion_dict = {'vp': 0, 'vn': 0, 'fp': 0, 'fn': 0}
    #     for i, label in enumerate(labels):
    #         if label == predictions[i]:
    #             if label == 1:
    #                 confusion_dict['vp'] += 1
    #             else:
    #                 confusion_dict['vn'] += 1
    #         else:
    #             if label == 1 and predictions[i] == 0:
    #                 confusion_dict['fn'] += 1
    #             elif label == 0 and predictions[i] == 1:
    #                 confusion_dict['fp'] += 1
    #     return confusion_dict

    def fit_and_evaluate(self, epochs, batch_size):
        """Nani."""
        model = self.model.fit(self.train_x, self.train_y, batch_size=batch_size,
                               epochs=epochs, verbose=0,
                               use_multiprocessing=True)
        history_train = model.history['loss']

        _, acc, f1_score, precision, recall = self.model.evaluate(self.test_x, self.test_y,
                                                                  batch_size=batch_size, verbose=0,
                                                                  use_multiprocessing=True)
        preds = self.model.predict(self.test_x, verbose=1,
                                   use_multiprocessing=True)
        preds = [int(round(pred[0])) for pred in preds]
        conf_matrix = [self.test_y, preds]
        self.log('Test Accuracy:' + str(acc))
        return {'acc': acc, 'loss': history_train,
                'status': STATUS_OK, 'model': self.model,
                'cm': conf_matrix, 'f1_score': f1_score, 'precision': precision, 'recall': recall}

    @staticmethod
    def log(message):
        """Nani."""
        print('[NeuralNetwork] ' + message)
