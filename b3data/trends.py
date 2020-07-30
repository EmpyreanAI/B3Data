import pandas
from b3data.stocks import Stocks
from pkg_resources import resource_filename, Requirement

class Trend:
    
    def __init__(self, code):
        self.code = code
        self._get_trend(code)

    def _get_trend(self, code):
        self.df = pandas.read_csv(resource_filename(Requirement.parse("b3data"), 'data/trends/' + code + '.csv'))
        self.df.rename(columns={ self.df.columns[1]: "trend" }, inplace=True)
        self.df['Week'] = self.df['Week'].astype('datetime64[ns]')

    def trends(self, year=2014, period=6):
        stock = Stocks(self.code, period=period)

        trend_list = []
        begin = True
        i = 0

        stock_df = stock.quotations
        stock_df['index'] = list(range(len(stock_df)))
        stock_df.set_index('index', inplace=True)

        for index in range(len(stock_df)-1):
            if begin and stock_df['DATA'][index] <= self.df['Week'][i]:
                trend_list.append(0)
            elif stock_df['DATA'][index] <= self.df['Week'][i+1]:
                begin = False
                trend_list.append(self.df['trend'][i])
            else:
                begin = False
                i += 1
                trend_list.append(self.df['trend'][i])
        
        return trend_list