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
fields = [CLOSING]


def run():
    """Nani."""
    results_acc = []
    results_f1_score = []
    df = pandas.DataFrame()
    df_f1_score = pandas.DataFrame()
    cod = sys.argv[1]
    batch_size = int(sys.argv[2])
    look_back = int(sys.argv[3])
    optimizer = sys.argv[4]
    stocks = Stocks(year=2014, cod=cod, period=5)
    dataset = stocks.selected_fields(copy(fields))
    dataset = duplicate_data(dataset)

    for cell in tqdm(cells):
        cells_results = []
        cells_f1_score_results = []
        for i in range(10):
            sequencial_kfold = SequencialKFold(n_split=6)
            acc, f1_score, loss, conf_mat =                          sequencial_kfold.split_and_fit(data=dataset,
                                           cells=cell,
                                           optimizer=optimizer,
                                           batch_size=batch_size,
                                           look_back=look_back)
            results_acc.append(acc)
            results_f1_score.append(f1_score)
            cells_results.append(numpy.mean(acc))
            cells_f1_score_results.append(numpy.mean(f1_score))

        df_temp = pandas.DataFrame({'cells_' + str(cell): cells_results})
        df_f1_score_temp = pandas.DataFrame({'cells_' + str(cell): cells_f1_score_results})
        df = pandas.concat([df, df_temp], axis=1)
        df_f1_score = pandas.concat([df_f1_score, df_f1_score_temp], axis=1)

    outname = 'posthoc_' + cod + '.csv'
    outname_f1_score = 'posthoc_f1_score_' + cod + '.csv'
    outdir = '../results'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    fullname = os.path.join(outdir, outname)
    fullname_f1_score = os.path.join(outdir, outname_f1_score)
    df.to_csv(fullname, mode='a')
    df_f1_score.to_csv(fullname_f1_score, mode='a')

    return results_acc

results = run()
