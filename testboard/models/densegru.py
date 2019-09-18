"""Nani."""

import numpy
from keras.models import Sequential
from keras.layers import GRU, Dense
from sklearn.preprocessing import MinMaxScaler

from .neuralnetwork import NeuralNetwork


class DenseGRU(NeuralNetwork):
    """Class responsible for create keras LSTM.

    It can be Dense with several LSTM cells or a single LSTM cell
    without Dense Layer.
    """

    def __init__(self, look_back=12, dense=True,
                 gru_cells=1000, input_shape=1):
        """Nani."""
        self.look_back = look_back
        self.dense = dense
        self.gru_cells = gru_cells
        self.input_shape = input_shape
        super(DenseGRU, self).__init__()

    def _create_model(self):
        """Nani."""
        model = Sequential()
        gru_cells = 1 if not self.dense else self.gru_cells
        model.add(GRU(gru_cells, input_shape=(self.input_shape,
                                              self.look_back)))
        if self.dense:
            model.add(Dense(1))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['acc'])

        return model
