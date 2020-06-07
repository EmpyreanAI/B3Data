# Treinar com todos os anos e testar em 2019
# Treinar um ano e testar no próximo
# Treinar modelos por trimestre/bimestre/semestre por ano
# Repetir os testes com label de proximo dia e de media

import sys
sys.path.append('../')

import numpy
from testboard.data_mining.stocks import Stocks
from testboard.data_mining.stocks import CLOSING
from testboard.validators.sequecialkfold import SequencialKFold
from testboard.models.denselstm import DenseLSTM
from keras import backend as K
import pandas
from tqdm import tqdm

models = [
    {'code': 'PETR3', 'window_size': 9, 'batch_size': 2, 'lstm_units': 80, 'optimizer': 'rmsprop'},
    {'code': 'VALE3', 'window_size': 6, 'batch_size': 2, 'lstm_units': 50, 'optimizer': 'rmsprop'},
    {'code': 'ABEV3', 'window_size': 6, 'batch_size': 128, 'lstm_units': 1, 'optimizer': 'adam'}
]

years = numpy.arange(2014, 2018)
periods = [5, 11]


for period in periods:
    df = pandas.DataFrame(columns=['PETR3', 'ABEV3', 'VALE3'], index=years)
    for model in tqdm(models):
        for year in years:
            stocks = Stocks(year=year, cod=model['code'], period=period)
            dataset = stocks.selected_fields([CLOSING])
            print(dataset)
            denselstm = DenseLSTM(input_shape=dataset.shape[1],
                                  look_back=model['window_size'],
                                  lstm_cells=model['lstm_units'],
                                  optimizer=model['optimizer'])
            denselstm.create_data_for_fit(dataset)
            result = denselstm.fit_and_evaluate(epochs=5000, batch_size=model['batch_size'])

            df.loc[year, [model['code']]] = result['acc']
            print(df)
            K.clear_session()

    df.to_csv('../results/period_' + str(period) + ".csv")
