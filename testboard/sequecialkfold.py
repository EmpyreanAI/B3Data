"""Nani."""

from denselstm import DenseLSTM
from densegru import DenseGRU
from keras import backend as K


class SequencialKFold():
    """Nani."""

    def __init__(self, n_split=10):
        """Nani."""
        self.n_split = n_split

    # Tem que refatorar ainda
    def split_and_fit(self, data=None, epochs=5000, look_back=0.25):
        """Nani."""
        acc_list = []

        if data is None:
            self.log("data paramater can't be None")
        else:
            data_len = len(data)
            jump_size = int(data_len/self.n_split)
            model = {}
            for i in range(1, self.n_split+1):
                data_splited = data[:jump_size*i, :]

                new_look_back = (len(data_splited)*0.3)*look_back
                self.log('LOOK_BACK = ' + str(int(new_look_back)))
                self.log('Data_Size = ' + str(int(len(data_splited))))
                self.log('PROPORTION = ' + str(new_look_back))
                self.log('DATA_SHAPE = ' + str(data_splited.shape))
                del model
                model = DenseLSTM(input_shape=data_splited.shape[1],
                                  look_back=int(new_look_back))
                model.create_data_for_fit(data_splited)
                result = model.fit_and_evaluate(epochs=epochs)
                acc_list.append(result['acc'])
                K.clear_session()

        return acc_list

    @staticmethod
    def log(message):
        """Nani."""
        print('[SequencialKFold] ' + message)
