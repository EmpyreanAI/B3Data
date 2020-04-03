"""Nani."""

from keras.models import Sequential
from keras.layers import GRU, Dense
from keras import backend as K

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

        def recall_m(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall

        def precision_m(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            return precision

        def f1_m(y_true, y_pred):
            precision = precision_m(y_true, y_pred)
            recall = recall_m(y_true, y_pred)
            return 2*((precision*recall)/(precision+recall+K.epsilon()))

        model = Sequential()
        gru_cells = 1 if not self.dense else self.gru_cells
        model.add(GRU(gru_cells, input_shape=(self.input_shape,
                                              self.look_back)))
        if self.dense:
            model.add(Dense(1))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['acc', f1_m, precision_m, recall_m])

        return model
