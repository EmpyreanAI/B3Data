"""Nani."""

import gc
import itertools
from copy import copy
from sequecialkfold import SequencialKFold

from stocks import Stocks
from stocks import CLOSING, OPENING, MAX_PRICE, MIN_PRICE, MEAN_PRICE, VOLUME
from plotter import Plotter
from smote import duplicate_data


class Experimenter():
    """Nani."""

    def __init__(self):
        """Nani."""
        self.plotter = Plotter()
        self.years = [2014, 2015, 2016, 2017]
        # self.years = [2017]
        # all_fields = [CLOSING, OPENING, MAX_PRICE,
                      # MIN_PRICE, MEAN_PRICE, VOLUME]
        # self.fields = []
        # self.stocks = ['VALE3', 'PETR3', 'ABEV3']
        self.stocks = ['PETR3', 'ABEV3']

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
            name += i[:5]
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
                    data_acc, data_loss = self.execute_experiment(year,
                                                                  stock,
                                                                  copy(field))
                    # self.plotter.acc_box_plot(data_acc, stock, year, s_field)
                    # self.plotter.loss_epoch_plot(data_loss, stock, year)
                    self.plotter.loss_acc_plot(data_acc, data_loss,
                                            stock, year, s_field)
                    gc.collect()

    @staticmethod
    def execute_experiment(year, stock, fields):
        """Nani."""
        results_acc = []
        results_loss = []

        stocks = Stocks(year=year, cod=stock, period=6)
        dataset = stocks.selected_fields(fields)
        dataset = duplicate_data(dataset)
        sequencial_kfold = SequencialKFold(n_split=10)
        for i in [0.25, 0.50, 0.75, 1]:
            acc, loss = sequencial_kfold.split_and_fit(data=dataset,
                                                       look_back=i)
            results_acc.append(acc)
            results_loss.append(loss)

        return results_acc, results_loss

    @staticmethod
    def log(message):
        """Nani."""
        print('[Experimenter] ' + message)
