"""Nani."""

import gc
from copy import copy

from validators.sequecialkfold import SequencialKFold
from data_mining.stocks import Stocks, CLOSING, OPENING, MAX_PRICE
from data_mining.stocks import MIN_PRICE, MEAN_PRICE, VOLUME
from simulator.plotter import Plotter
from data_mining.smote import duplicate_data


class Experimenter():
    """Nani."""

    def __init__(self):
        """Nani."""
        self.plotter = Plotter()
        self.years = [2014, 2015, 2016, 2017]
        # self.years = [2017]
        # all_fields = [CLOSING, OPENING, MAX_PRICE,
        #               MIN_PRICE, MEAN_PRICE, VOLUME]
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
                    data_acc, data_loss, conf_mats, data_amts = self.execute_experiment(year,
                                                                             stock,
                                                                             copy(field))
                    look_backs = [0.25, 0.5, 0.75, 1.0]
                    for conf_mat, lb in zip(conf_mats, look_backs):
                        self.plotter.plot_confusion_matrix(conf_mat[0],
                                                           conf_mat[1],
                                                           stock, year,
                                                           s_field, lb)
                    self.plotter.loss_acc_plot(data_acc, data_loss,
                                               stock, year, s_field, data_amts)
                    gc.collect()

    @staticmethod
    def execute_experiment(year, stock, fields):
        """Nani."""
        results_acc = []
        results_loss = []
        conf_mats = []
        data_amts = []

        stocks = Stocks(year=year, cod=stock, period=5)
        dataset = stocks.selected_fields(fields)
        dataset = duplicate_data(dataset)
        sequencial_kfold = SequencialKFold(n_split=10)
        for i in [0.25, 0.50, 0.75, 1]:
            acc, loss, conf_mat, data_amt = sequencial_kfold.split_and_fit(data=dataset,
                                                                           look_back=i)
            conf_mats.append(conf_mat)
            data_amts.append(data_amt)
            results_acc.append(acc)
            results_loss.append(loss)

        return results_acc, results_loss, conf_mats, data_amts

    @staticmethod
    def log(message):
        """Nani."""
        print('[Experimenter] ' + message)
