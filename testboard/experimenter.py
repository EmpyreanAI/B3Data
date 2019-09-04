import itertools
from plotter import Plotter
from denselstm import DenseLSTM
from stocks import Stocks, OPENING, CLOSING, MEAN_PRICE, MIN_PRICE, MAX_PRICE, VOLUME

class Experimenter(object):

    def __init__(self):
        self.plotter = Plotter()
        self.stocks = ['PETR3', 'ABEV3', 'VALE3', 'BBAS3', 'BVMF3']
        self.years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
        all_fields = [OPENING, CLOSING, MEAN_PRICE, MIN_PRICE, MAX_PRICE, VOLUME]
        self.fields = []

        for field in range(0, len(all_fields)+1):
            for subset in itertools.combinations(all_fields, field):
                if len(subset) != 0:
                    self.fields.append(list(subset))


    def run(self):
        for stock in self.stocks:
            for year in self.years:
                for field in self.fields:
                    data = self.execute_experiment(year, stock, field)
                    self.plotter.box_plot(data, stock, year)

    def execute_experiment(self, year, stock, fields):
        stocks = Stocks(year=year, cod=stock)
        dataset = stocks.selected_fields(fields)
        lstm = DenseLSTM(input_shape=dataset.shape[1], look_back=30)
        lstm.create_data_for_fit(dataset, mean_of=0)
        lstm.fit_and_evaluate(epochs=10000)
        # MUDAR ISSO AQUI TAOKEY
        data = []
        return data

# exp = Experimenter()
