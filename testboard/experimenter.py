"""Nani."""

import gc
import itertools
from copy import copy
from sequecialkfold import SequencialKFold

from stocks import Stocks
from stocks import CLOSING, OPENING, MAX_PRICE, MIN_PRICE, MEAN_PRICE, VOLUME
from plotter import Plotter


class Experimenter():
    """Nani."""

    def __init__(self):
        """Nani."""
        self.plotter = Plotter()
        self.years = [2014, 2015, 2016, 2017]
        # all_fields = [CLOSING, OPENING, MAX_PRICE,
                      # MIN_PRICE, MEAN_PRICE, VOLUME]
        # self.fields = []
        self.stocks = ['VALE3', 'PETR3', 'ABEV3']

        # for i, _ in enumerate(all_fields):
        #     for subset in itertools.combinations(all_fields, i):
        #         if subset:
        #             self.fields.append(list(subset))
        self.fields = [[CLOSING], [OPENING], [MAX_PRICE],
                      [MIN_PRICE], [MEAN_PRICE], [VOLUME]]

    @staticmethod
    def gen_str_fields(field):
        """Nani."""
        name = ''
        for i in field:
            name += i[:4]
            name += '_'
        return name

    def run(self):
        """Nani."""
        for stock in self.stocks:
            for year in self.years:
                for field in self.fields:
                    s_field = self.gen_str_fields(field)
                    self.log("Stock: %s; Year: %s; Fields: %s"
                              % (stock, str(year), s_field))
                    data = self.execute_experiment(year, stock, copy(field))
                    self.plotter.box_plot(data, stock, year, s_field)
                    gc.collect()

    @staticmethod
    def execute_experiment(year, stock, fields):
        """Nani."""
        results = []
        stocks = Stocks(year=year, cod=stock, period=11)
        dataset = stocks.selected_fields(fields)
        sequencial_kfold = SequencialKFold(n_split=10)
        for i in [0.25, 0.50, 0.75, 1]:
            res = sequencial_kfold.split_and_fit(data=dataset,
                                                 look_back=i)
            results.append(res)

        return results

    @staticmethod
    def log(message):
        """Nani."""
        print('[Experimenter] ' + message)
