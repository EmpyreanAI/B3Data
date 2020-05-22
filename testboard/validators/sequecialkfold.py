"""Nani."""
import sys
sys.path.append('../testboard')

from models.denselstm import DenseLSTM

import numpy as np
from keras import backend as K


class SequencialKFold():
    """Nani."""

    def __init__(self, n_split=10):
        """Nani."""
        self.n_split = n_split

    # Tem que refatorar ainda
    def split_and_fit(self, data=None, epochs=5000, look_back=12, cells=1,
                      batch_size=1,
                      optimizer='rmsprop'):
        """Nani."""
        acc_list = []
        fscore_list = []
        loss_list = []
        conf_dict = [[], []]

        if data is None:
            self.log("data paramater can't be None")
        else:
            data_len = len(data)
            jump_size = int(data_len/self.n_split)
            model = {}
            for i in range(1, self.n_split+1):
                data_splited = data[:jump_size*i, :]

                self.log('LOOK_BACK = ' + str(look_back))
                self.log('Data_Size = ' + str(int(len(data_splited))))
                self.log('DATA_SHAPE = ' + str(data_splited.shape))
                del model
                print(data_splited)
                model = DenseLSTM(input_shape=data_splited.shape[1],
                                  look_back=look_back, lstm_cells=cells,
                                  optimizer=optimizer)
                model.create_data_for_fit(data_splited)
                result = model.fit_and_evaluate(epochs=epochs, batch_size=batch_size)
                acc_list.append(result['acc'])
                fscore_list.append(result['f1_score'])
                loss_list.append(result['loss'])
                conf_dict[0] += list(result['cm'][0])
                conf_dict[1] += list(result['cm'][1])
                K.clear_session()

            mean_list = np.mean(loss_list, axis=0)

        return acc_list, fscore_list, mean_list, conf_dict

    @staticmethod
    def log(message):
        """Nani."""
        print('[SequencialKFold] ' + message)
