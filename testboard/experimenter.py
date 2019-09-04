from denselstm import DenseLSTM
from stocks import Stocks, OPENING, CLOSING, MEAN_PRICE, MIN_PRICE, MAX_PRICE, VOLUME

class Experimenter(object):

    def __init__(self):
        self.stocks = ['PETR3']
        self.years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
        self.fields = [CLOSING, OPENING, VOLUME]

    def run(self):
        for stock in self.stocks:
            for year in self.years:
                for field in self.fields:
                    self.execute_experiment(year, stock, field)

    def execute_experiment(self, year, stock, fields):
        stocks = Stocks(year=year, cod=stock)
        dataset = stocks.selected_fields(fields)
        lstm = DenseLSTM(input_shape=dataset.shape[1], look_back=30)
        lstm.create_data_for_fit(dataset, mean_of=0)
        lstm.fit_and_evaluate(epochs=10000)
