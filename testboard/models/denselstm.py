"""Nani."""

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras import backend as K

from .neuralnetwork import NeuralNetwork


class DenseLSTM(NeuralNetwork):
    """Class responsible for create keras LSTM.

    It can be Dense with several LSTM cells or a single LSTM cell
    without Dense Layer.
    """

    def __init__(self, look_back=12, dense=True,
                 lstm_cells=1, input_shape=1, optimizer='rmsprop'):
        """Nani."""
        self.look_back = look_back
        self.dense = dense
        self.lstm_cells = lstm_cells
        self.input_shape = input_shape
        self.optimizer = optimizer
        super(DenseLSTM, self).__init__()

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
        lstm_cells = 1 if not self.dense else self.lstm_cells

        model.add(LSTM(lstm_cells,
                       input_shape=(self.input_shape, self.look_back),
                       bias_initializer='random_normal'))

        if self.dense:
            model.add(Dense(activation="sigmoid", units=1))

        model.compile(loss='binary_crossentropy',
                      optimizer=self.optimizer,
                      metrics=['accuracy', f1_m, precision_m, recall_m])

        # model.summary()

        return model
