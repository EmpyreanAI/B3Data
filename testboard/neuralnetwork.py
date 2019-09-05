from hyperopt import Trials, STATUS_OK, tpe

class NeuralNetwork(object):

  # @staticmethod
  def __create_model(self):
    pass


  def create_data_for_fit():
    pass

  def fit_and_evaluate(self, epochs=5000):
    self.model.fit(self.trainX, self.trainY, epochs, verbose=0, validation_split=0.33)
    score, acc = self.model.evaluate(self.testX, self.testY, verbose=0)
    self.log('Test accuracy:' + str(acc))
    return {'acc': acc, 'status': STATUS_OK, 'model': self.model}

  @staticmethod
  def log(message):
    print('[NeuralNetwork] ' + message)
