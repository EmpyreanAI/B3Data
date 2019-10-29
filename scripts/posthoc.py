import sys
sys.path.append('../')
sys.path.append('../testboard')

from testboard.validators.sequecialkfold import SequencialKFold
from testboard.data_mining.stocks import Stocks, CLOSING, OPENING, MAX_PRICE
from testboard.data_mining.stocks import MIN_PRICE, MEAN_PRICE, VOLUME
from testboard.simulator.plotter import Plotter
from testboard.data_mining.smote import duplicate_data
import pandas
import numpy
import os
from tqdm import tqdm
from copy import copy

plotter = Plotter()
cells = [1, 50, 80, 100, 150, 200]
years = [2014, 2015, 2016, 2017]
fields = [CLOSING]


def run():
    """Nani."""
    results_acc = []
    df = pandas.DataFrame()
    stocks = Stocks(year=2014, cod='VALE3', period=5)
    dataset = stocks.selected_fields(copy(fields))
    dataset = duplicate_data(dataset)

    for cell in tqdm(cells):
        cells_results = []
        for i in range(10):
            sequencial_kfold = SequencialKFold(n_split=6)
            acc, loss, conf_mat = sequencial_kfold.split_and_fit(data=dataset,
                                                                 cells=cell)
            results_acc.append(acc)
            cells_results.append(numpy.mean(acc))

        df_temp = pandas.DataFrame({'cells_' + str(cell): cells_results})
        df = pandas.concat([df, df_temp], axis=1)

    outname = 'cells_experiment_VALE.csv'
    outdir = '../results'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    fullname = os.path.join(outdir, outname)
    df.to_csv(fullname, mode='a')

    return results_acc

results = run()
