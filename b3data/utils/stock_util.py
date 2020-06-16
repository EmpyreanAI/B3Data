import sys
sys.path.append('../../')

from b3data.stocks import Stocks, CLOSING
from b3data.utils.smote import duplicate_data

class StockUtil(object):

    def __init__(self, stocks, windows):
        self.stocks = stocks
        self.windows = windows

    def prices_preds(self, start_year=2014, end_year=2016, period=11):

        small_dataset = float('inf')
        prices = []
        preds = []

        for i in range(len(self.stocks)):
            cod = self.stocks[i]
            win = self.windows[i]

            s_prices = Stocks.interval_of_years(cod, start_year, end_year, 1, period=period)
            s_prices = s_prices.reshape(1, len(s_prices))[0]
            s_preds = [1 if s_prices[i] >= s_prices[i-win:i].mean() else 0 for i in range(win, len(s_prices))]
            s_prices = s_prices[win:]
            small_dataset = min(small_dataset, len(s_prices))
            prices.append(s_prices)
            preds.append(s_preds)

        for i in range(len(self.stocks)):
            prices[i] = prices[i][:small_dataset-1]
            preds[i] = preds[i][:small_dataset-1]

        return prices, preds

    def average_prices_preds(self, year=2014, period=5):

        small_dataset = float('inf')
        prices = []
        preds = []

        for i in range(len(self.stocks)):
            cod = self.stocks[i]
            win = self.windows[i]

            stock = Stocks(year=year, cod=cod, period=period)
            s_prices = stock.selected_fields([CLOSING])
            s_prices = duplicate_data(s_prices)
            aux_prices = s_prices.reshape(1, len(s_prices))[0]
            s_preds = [1 if aux_prices[i] >= aux_prices[i-win:i].mean() else 0 for i in range(win, len(s_prices))]
            s_prices = [aux_prices[i-win:i].mean() for i in range(win, len(aux_prices))]
            small_dataset = min(small_dataset, len(s_prices))
            prices.append(s_prices)
            preds.append(s_preds)

        for i in range(len(self.stocks)):
            prices[i] = prices[i][:small_dataset-1]
            preds[i] = preds[i][:small_dataset-1]

        return prices, preds
