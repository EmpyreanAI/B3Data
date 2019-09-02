from denselstm import DenseLSTM
import neuralnetwork
from stocks import Stocks, OPENING, CLOSING, MEAN_PRICE, MIN_PRICE, MAX_PRICE, VOLUME

if __name__ == '__main__':
    stocks = Stocks(year=2015, cod='PETR3')
    dataset = stocks.selected_fields([CLOSING, OPENING, VOLUME])
    lstm = DenseLSTM(input_shape=dataset.shape[1], look_back=30)
    lstm.create_data_for_fit(dataset, mean_of=0)
    lstm.fit_and_evaluate(epochs=10000)
