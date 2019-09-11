"""Nani."""

from hyperopt import STATUS_OK


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
        self.model.fit(self.train_x, self.train_y, epochs,
                       verbose=0, validation_split=0.33)
        _, acc = self.model.evaluate(self.test_x, self.test_y,
                                     verbose=0)
        self.log('Test accuracy:' + str(acc))
        return {'acc': acc, 'status': STATUS_OK, 'model': self.model}

    @staticmethod
    def log(message):
        """Nani."""
        print('[NeuralNetwork] ' + message)
