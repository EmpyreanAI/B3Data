import glob
import numpy
import pandas as pd

CLOSING = 'PREULT'
OPENING = 'PREABE'
MAX_PRICE = 'PREMAX'
MIN_PRICE = 'PREMAX'
MEAN_PRICE = 'PREMED'
VOLUME = 'VOLTOT'

class Stocks(object):

  def __init__(self, cod='PETR3', year=2014, start_month=1, period=0):
    path = "../BovespaWolf/data/COTAHIST_A" + str(year) + ".TXT.csv"
    files = glob.glob(path)

    for name in files:
      with open(name) as file:
        df = pd.read_csv(file, low_memory=False)

      df = df.drop(df.index[-1])
      df = self.__changing_column_data_type(df)
      filtered = df.loc[df['CODNEG'] == cod]

      if (start_month + period) > 12:
        next_month = 12
        self.log("Interval time surpassed the last month. Last month will be 12")
      else:
        next_month = start_month + period

      first_date = str(year) + '-0' + str(start_month) + '-01'
      last_date = str(year) + '-0' + str(next_month) + '-30'
      stocks = filtered.loc[(filtered['DATA'] > first_date) &
                 (filtered['DATA'] <= last_date)]
                 
      self.quotations = stocks

  def selected_fields(self, fields=[CLOSING]):
    if not fields:
      self.log('Argument fields cant be null')
    else:
      selected = self.quotations[[fields[0]]].values
      fields.pop(0)
      for i in fields:
         selected = numpy.column_stack((selected, self.quotations[[i]].values))

    return selected


  @staticmethod
  def log(message):
    print("[Stocks] " + message)

  def __changing_column_data_type(self, df):
    df['DATA'] = df['DATA'].astype(str)
    df['CODNEG'] = df['CODNEG'].astype(str)
    df['PREULT'] = (df['PREULT']/100).astype(float)
    df['PREABE'] = (df['PREABE']/100).astype(float)
    df['PREMAX'] = (df['PREMAX']/100).astype(float)
    df['PREMIN'] = (df['PREMIN']/100).astype(float)
    df['PREMED'] = (df['PREMED']/100).astype(float)
    df['VOLTOT'] = (df['VOLTOT']/100).astype(float)
    df['DATA'] = pd.to_datetime(df['DATA'],
                    format='%Y%m%d',
                    errors='ignore')
    return df
