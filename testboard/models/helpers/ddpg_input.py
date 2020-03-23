import sys
sys.path.append('../../')

from data_mining.stocks import Stocks, CLOSING

class DDPGInput(object):

    def __init__(self, stocks, windows):
        self.stocks = stocks
        self.windows = windows

    def prices_preds(self, year=2014, period=5):

        small_dataset = float('inf')
        prices = []
        preds = []

        for i in range(len(self.stocks)):
            cod = self.stocks[i]
            win = self.windows[i]

            stock = Stocks(year=year, cod=cod, period=period)
            s_prices = stock.selected_fields([CLOSING])
            s_prices = s_prices.reshape(1, len(s_prices))[0]
            s_preds = [1 if s_prices[i+1] >= s_prices[:i-win].mean() else 0 for i in range(win, len(s_prices)-1)]
            s_prices = s_prices[win:]
            small_dataset = min(small_dataset, len(s_prices))
            prices.append(s_prices)
            preds.append(s_preds)

        for i in range(len(self.stocks)):
            prices[i] = prices[i][:small_dataset-1]
            preds[i] = preds[i][:small_dataset-1]

        return prices, preds
