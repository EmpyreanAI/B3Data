import pandas as pd
import glob
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def prices(year, cod, month=1, period=1):
    path = "../BovespaWolf/data/COTAHIST_A" + str(year) + ".TXT.csv"
    files = glob.glob(path)

    for name in files:
        print(name)
        with open(name) as file:
            df = pd.read_csv(file, low_memory=False)

        df = df.drop(df.index[-1])

        df['DATA'] = df['DATA'].astype(str)
        df['TIPREG'] = df['TIPREG'].astype(int)
        df['CODBDI'] = df['CODBDI'].astype(int)
        df['CODNEG'] = df['CODNEG'].astype(str)


        df['TPMERC'] = df['TPMERC'].astype(int)

        df['NOMRES'] = df['NOMRES'].astype(str)
        df['ESPECI'] = df['ESPECI'].astype(str)
        df['PRAZOT'] = df['PRAZOT'].astype(str)
        df['MODREF'] = df['MODREF'].astype(str)


        df['PREABE'] = (df['PREABE']/100).astype(float)
        df['PREMAX'] = (df['PREMAX']/100).astype(float)
        df['PREMIN'] = (df['PREMIN']/100).astype(float)
        df['PREMED'] = (df['PREMED']/100).astype(float)
        df['PREULT'] = (df['PREULT']/100).astype(float)
        df['PREOFC'] = (df['PREOFC']/100).astype(float)
        df['PREOFV'] = (df['PREOFV']/100).astype(float)

        df['TOTNEG'] = df['TOTNEG'].astype(int)
        df['QUATOT'] = df['QUATOT'].astype(np.int64)
        df['VOLTOT'] = df['VOLTOT'].astype(np.int64)
        df['PREEXE'] = (df['PREEXE']/100).astype(float)
        df['INDOPC'] = df['INDOPC'].astype(int)
        df['DATAEN'] = df['DATAEN'].astype(str)
        df['FATCOT'] = df['FATCOT'].astype(int)
        df['PTOEXE'] = df['PTOEXE'].astype(np.int64)
        df['CODISI'] = df['CODISI'].astype(str)
        df['DISMES'] = df['DISMES'].astype(int)

        df['DATA'] = pd.to_datetime(df['DATA'],
                                        format='%Y%m%d',
                                        errors='ignore')

        filtered = df.loc[df['CODNEG'] == cod]

        next_month = (month + period) % 12

        stocks = filtered.loc[(filtered['DATA'] > str(year) + '-' + str(month) + '-01') & 
                              (filtered['DATA'] <= str(year) + '-' + str(next_month) + '-01')]
        
        return stocks['PREULT']

def make_direction_array(data):
    directions = []
    for i in range(len(data)-1):
        if data[i] < data[i+1]:
            directions.append(1)
        else:
            directions.append(0)

    return directions

def score(directions, predict):
    score = []
    for i in range(len(directions)):
        score.append(directions[i] == predict[i])

    return sum(score)/len(directions)

stock_name = 'PETR3'
year = 2014
periods = [1, 2, 3, 4, 5, 6]
month = 1
alphas = [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
betas = [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]

df = pd.DataFrame()

for period in periods:

    data = prices(year, stock_name, month, period)

    for alpha in alphas:

        model = SimpleExpSmoothing(list(data))
        model_fit = model.fit(alpha)

        yhat = model_fit.predict(start=0, end=len(data))

        true_directions = make_direction_array(list(data))
        predictions = make_direction_array(yhat)
        
        accuracy = score(true_directions, predictions)

        register = {'period': period, 'alpha':alpha, 'accuracy': accuracy}

        df = df.append(register, ignore_index=True)


df.to_csv('./results/ses_experiments.csv', header=True, index=False)

# fig = plt.figure()
# plt.plot(yhat, label="ses")
# plt.plot(list(data), label=stock_name)
# plt.legend(loc='upper right')
# fig.suptitle(str(year) + "/" + 
#              str(month) + " " +  
#              stock_name + " alpha" + 
#              str(alpha) + "p" + str(period))
# plt.show()
# fig.savefig("images/" + str(year) + "-" + 
#             str(month) + stock_name + 
#             "alpha" + str(alpha) + "p" + 
#             str(period) + ".png")