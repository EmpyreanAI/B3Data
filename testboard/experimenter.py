import itertools
import gc
from plotter import Plotter
from copy import copy
from denselstm import DenseLSTM
from sequecialkfold import SequencialKFold
from stocks import Stocks, OPENING, CLOSING, MEAN_PRICE, MIN_PRICE, MAX_PRICE, VOLUME

class Experimenter(object):

  def __init__(self):
    self.plotter = Plotter()
    self.stocks = ['PETR3', 'ABEV3', 'VALE3', 'BBAS3', 'BVMF3']
    self.years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
    all_fields = [CLOSING]
    self.fields = []

    for field in range(0, len(all_fields)+1):
        for subset in itertools.combinations(all_fields, field):
            if len(subset) != 0:
                self.fields.append(list(subset))

  def generate_string_fields(self, field):
    name = ''
    for i in field:
      name += i[:4]
      name += '_'

    return name

  def run(self):
    for stock in self.stocks:
      for year in self.years:
        for field in self.fields:
          self.log('stock: ' + str(stock) + ' year: ' + str(year) + ' fields: ' + self.generate_string_fields(field))
          data = self.execute_experiment(year, stock, copy(field))
          self.plotter.box_plot(data, stock, year, features=self.generate_string_fields(field))
          gc.collect()

  def execute_experiment(self, year, stock, fields):
    results = []
    stocks = Stocks(year=year, cod=stock, period=6)
    dataset = stocks.selected_fields(fields)
    sequencial_kfold = SequencialKFold(n_split=10)
    for i in [0.25, 0.50, 0.75, 1]:
      results.append(sequencial_kfold.split_and_fit(data=dataset, look_back_proportion=i))

    del stock
    return results

  def log(self, message):
    print('[Experimenter]' + message)

# exp = Experimenter()
