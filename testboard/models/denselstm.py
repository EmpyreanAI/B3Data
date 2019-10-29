"""Nani."""

from keras.models import Sequential
from keras.layers import LSTM, Dense

from .neuralnetwork import NeuralNetwork


class DenseLSTM(NeuralNetwork):
    """Class responsible for create keras LSTM.

    It can be Dense with several LSTM cells or a single LSTM cell
    without Dense Layer.
    """

    def __init__(self, look_back=12, dense=True,
                 lstm_cells=1, input_shape=1):
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

        model.add(LSTM(lstm_cells,
                       input_shape=(self.input_shape, self.look_back),
                       bias_initializer='random_normal'))

        if self.dense:
            model.add(Dense(activation="sigmoid", units=1))

        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

        return model
