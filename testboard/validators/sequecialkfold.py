"""Nani."""

from models.denselstm import DenseLSTM

import numpy as np
from keras import backend as K
from collections import Counter


class SequencialKFold():
    """Nani."""

    def __init__(self, n_split=10):
        """Nani."""
        self.n_split = n_split

    # Tem que refatorar ainda
    def split_and_fit(self, data=None, epochs=1, look_back=0.50, cells=1):
        """Nani."""
        acc_list = []
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
                new_look_back = (len(data_splited)*0.3)*look_back
                self.log("LOOK BACK = {}".format(look_back))
                self.log("DATA SPLITED SIZE = {}".format(len(data_splited)*0.3))
                self.log('LOOK_BACK = ' + str(int(new_look_back)))
                self.log('Data_Size = ' + str(int(len(data_splited))))
                self.log('PROPORTION = ' + str(new_look_back))
                self.log('DATA_SHAPE = ' + str(data_splited.shape))
                del model
                model = DenseLSTM(input_shape=data_splited.shape[1],
                                  look_back=int(new_look_back), lstm_cells=cells)
                model.create_data_for_fit(data_splited)
                result = model.fit_and_evaluate(epochs=epochs)
                acc_list.append(result['acc'])
                loss_list.append(result['loss'])
                conf_dict[0] += list(result['cm'][0])
                conf_dict[1] += list(result['cm'][1])
                K.clear_session()

            mean_list = np.mean(loss_list, axis=0)

        import pdb; pdb.set_trace()
        # print("DATA_LEN * LOOK_BACK = {}".format(data_len*look_back))
        # print("NOVO LOOK BACK = {}".format(new_look_back))
        return acc_list, mean_list, conf_dict, new_look_back

    @staticmethod
    def log(message):
        """Nani."""
        print('[SequencialKFold] ' + message)
