import sys
sys.path.append('../')
sys.path.append('../testboard')

from testboard.validators.sequecialkfold import SequencialKFold
from testboard.data_mining.stocks import Stocks, CLOSING, OPENING, MAX_PRICE
from testboard.data_mining.stocks import MIN_PRICE, MEAN_PRICE, VOLUME
from testboard.simulator.plotter import Plotter
from testboard.data_mining.smote import duplicate_data
import pandas

plotter = Plotter()
cells = [1, 50, 80, 100, 150, 200]
fields = [CLOSING]

stocks = Stocks(year=2014, cod=sys.argv[1], period=5)
dataset = stocks.selected_fields(fields)
dataset = duplicate_data(dataset)
optmizer = sys.argv[2]
batch_size = int(sys.argv[3])

def run():
    """Nani."""
    results_acc = []

    for cell in cells:
        sequencial_kfold = SequencialKFold(n_split=6)
        acc, loss, conf_mat = sequencial_kfold.split_and_fit(data=dataset,
                                                             cells=cell,
                                                             optimizer=optmizer,
                                                             batch_size=batch_size)
        results_acc.append(acc)
    return results_acc
results = run()

plotter.acc_cells_box_plot(data=results, stock=sys.argv[1], year='2014')
