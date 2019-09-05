from denselstm import DenseLSTM

class SequencialKFold(object):
  def __init__(self, n_split=10):
    self.n_split = n_split

  def split_and_fit(self, data=None, callback=None, epochs=10000, look_back_proportion=0.25):
    acc_list = []

    if data is None:
      self.log("data paramater can't be None")
    else:
      data_len = len(data)
      jump_size = int(data_len/self.n_split)
      model = {}
      for i in range(1, self.n_split+1):
        data_splited = data[:jump_size*i,:]

        # PRECISA REFATORAR
        look_back = (len(data_splited)*0.3)*look_back_proportion
        self.log('LOOK_BACK = ' + str(int(look_back)))
        self.log('Data_Size = ' + str(int(len(data_splited))))
        self.log('PROPORTION = ' + str(look_back_proportion))
        self.log('DATA_SHAPE = ' + str(data_splited.shape))
        del model
        model = DenseLSTM(input_shape=data_splited.shape[1], look_back=int(look_back))
        model.create_data_for_fit(data_splited, mean_of=0)
        result = model.fit_and_evaluate(epochs=epochs)
        acc_list.append(result['acc'])

      return acc_list

  def log(self, message):
    print('[SequencialKFold] ' + message)
