import glob
import pandas as pd

class Stocks(object):

    @staticmethod
    def prices(year=2014, cod='PETR4', month=1, period=11):
        path = "../BovespaWolf/data/COTAHIST_A" + str(year) + ".TXT.csv"
        files = glob.glob(path)

        for name in files:
            print(name)
            with open(name) as file:
                df = pd.read_csv(file, low_memory=False)

            df = df.drop(df.index[-1])

            df['DATA'] = df['DATA'].astype(str)
            df['CODNEG'] = df['CODNEG'].astype(str)
            df['PREULT'] = (df['PREULT']/100).astype(float)
            df['DATA'] = pd.to_datetime(df['DATA'],
                                            format='%Y%m%d',
                                            errors='ignore')

            filtered = df.loc[df['CODNEG'] == cod]

            next_month = (month + period) % 13

            stocks = filtered.loc[(filtered['DATA'] > str(year) + '-' + str(month) + '-01') &
                                  (filtered['DATA'] <= str(year) + '-' + str(next_month) + '-30')]

            return stocks['PREULT']