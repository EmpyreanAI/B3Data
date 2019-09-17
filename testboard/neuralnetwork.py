"""Nani."""

from hyperopt import STATUS_OK
from callbacks import LossHistory
from plotter import Plotter


class NeuralNetwork():
    """Nani."""

    def __init__(self):
        """Nani."""
        self.train_x = []
        self.train_y = []
        self.test_x = []
        self.test_y = []
        self.model = self._create_model()

    def _create_model(self):
        """Nani."""
        raise NotImplementedError("Model must create an object.")

    def create_data_for_fit(self, dataset, train_proportion=0.66):
        """Nani."""
        raise NotImplementedError("Model must create data.")

    def fit_and_evaluate(self, epochs):
        """Nani."""
        history_train = LossHistory()
        self.model.fit(self.train_x, self.train_y, batch_size=256,
                       epochs=epochs, verbose=1, callbacks=[history_train])

        loss, acc = self.model.evaluate(self.test_x, self.test_y,
                                        batch_size=256, verbose=0)
        # Plotter.loss_epoch_plot(history_train.losses)
        self.log('Test Loss:' + str(loss))
        self.log('Test Accuracy:' + str(acc))
        return {'acc': acc, 'loss': history_train.losses,
                'status': STATUS_OK, 'model': self.model}

    @staticmethod
    def log(message):
        """Nani."""
        print('[NeuralNetwork] ' + message)
