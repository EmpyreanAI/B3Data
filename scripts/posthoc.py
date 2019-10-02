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

plotter = Plotter()
cells = [1, 50, 80, 100, 150, 200]
years = [2014, 2015, 2016, 2017]
fields = [CLOSING]


def run():
    """Nani."""
    results_acc = []
    df = pandas.DataFrame()

    for cell in tqdm(cells):
        cells_results = []
        for year in years:
            stocks = Stocks(year=year, cod='PETR3', period=6)
            dataset = stocks.selected_fields(fields)
            dataset = duplicate_data(dataset)
            sequencial_kfold = SequencialKFold(n_split=10)
            acc, loss, conf_mat = sequencial_kfold.split_and_fit(data=dataset,
                                                                 look_back=0.5,
                                                                 cells=cell)
            acc = [1,2,3,4,5,6,7]
            results_acc.append(acc)
            cells_results.append(numpy.mean(acc))

        df_temp = pandas.DataFrame({'cells_' + str(cell): cells_results})
        df = pandas.concat([df, df_temp], axis=1)

    outname = 'cells_experiment.csv'
    outdir = '../results'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    fullname = os.path.join(outdir, outname)
    df.to_csv(fullname, mode='a')

    return results_acc

results = run()

plotter.acc_cells_box_plot(data=results, stock='PETR3', year='2014')
