import pandas
from b3data.stocks import Stocks
from pkg_resources import resource_filename, Requirement
import datetime

class Trend:
    
    def __init__(self, code):
        self.code = code
        self._get_trend(code)

    def _get_trend(self, code):
        self.df = pandas.read_csv(resource_filename(Requirement.parse("b3data"), 'data/trends/' + code + '.csv'))
        self.df.rename(columns={ self.df.columns[1]: "trend" }, inplace=True)
        self.df['Week'] = self.df['Week'].astype('datetime64[ns]')

    def trends(self, start_month=1,  period=6):
        stock = Stocks(self.code, start_month=start_month, period=period)

        trend_list = []
        begin = True
        i = 0

        stock_df = stock.quotations
        stock_df['index'] = list(range(len(stock_df)))
        stock_df.set_index('index', inplace=True)

        l_bound, u_bound = self._get_lower_bound(stock_df)

        self.df = self.df[(self.df['Week'] >= l_bound) & \
                  (self.df['Week'] <= u_bound)]

        print(self.df)
        print(stock_df)

        for index in range(len(stock_df)):
            if i == len(self.df)-1:
                trend_list.append(self.df['trend'].iloc[-1])
            else:
                if stock_df['DATA'].iloc[index] <= datetime.datetime.strptime('2014-01-05', "%Y-%m-%d"):
                    trend_list.append(0)
                elif stock_df['DATA'].iloc[index] < self.df['Week'].iloc[i+1]:
                    trend_list.append(self.df['trend'].iloc[i])
                else:
                    i += 1
                    trend_list.append(self.df['trend'].iloc[i])

        print(len(trend_list))
        return trend_list

    def _get_lower_bound(self, stock_df):
        first_date = stock_df['DATA'].iloc[0]
        last_date = stock_df['DATA'].iloc[-1]
        lower_bound = first_date
        upper_bound = last_date
        for index, row in self.df.iterrows():
            if row['Week'] <= first_date:
                lower_bound = row['Week']
            if row['Week'] >= last_date:
                upper_bound = row['Week']

        return lower_bound, upper_bound


t = Trend('PETR3')
print(t.trends(start_month=7, period=6))