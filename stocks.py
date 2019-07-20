import glob
import pandas as pd

class Stocks(object):

  @classmethod
  def prices(cls, year=2014, cod='PETR4', start_month=1, period=11):
    path = "../BovespaWolf/data/COTAHIST_A" + str(year) + ".TXT.csv"
    files = glob.glob(path)

    for name in files:
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

      if (start_month + period) > 12:
        next_month = 12
        cls.log("Intervalo de tempo ultrapassou o último mês do ano. Último mês será considerado como o mês 12")
      else:
        next_month = start_month + period

      first_date = str(year) + '-' + str(start_month) + '-01'
      last_date = str(year) + '-' + str(next_month) + '-30'

      stocks = filtered.loc[(filtered['DATA'] > first_date) &
                 (filtered['DATA'] <= last_date)]

      return stocks['PREULT']

  @staticmethod
  def log(message):
    print("[Stocks]" + message)
