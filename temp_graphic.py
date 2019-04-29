import pandas
import glob
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

def prices(year, cod, month=1, period=1):
    path = "../BovespaWolf/data/COTAHIST_A" + str(year) + ".TXT.csv"
    files = glob.glob(path)

    for name in files:
        print(name)
        with open(name) as file:
            df = pandas.read_csv(file, low_memory=False)

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

        df['DATA'] = pandas.to_datetime(df['DATA'],
                                        format='%Y%m%d',
                                        errors='ignore')

        filtered = df.loc[df['CODNEG'] == cod]

        next_month = (month + period) % 12

        stocks = filtered.loc[(filtered['DATA'] > str(year) + '-' + str(month) + '-01') & 
                              (filtered['DATA'] <= str(year) + '-' + str(next_month) + '-01')]
        
        return stocks['PREULT']

def simple_exponencial_smoothing(data, alpha):
    output = [data[0]]
    for i in range(len(data)):
        output.append(alpha*data[i] + (1-alpha)*output[i])

    return output


stock_name = 'PETR3'
year = 2014
period = 6
month = 1
alpha = 0.5

data = prices(year, stock_name, month, period)

ses_predict = simple_exponencial_smoothing(list(data), alpha)

fig = plt.figure()
plt.plot(ses_predict, label="ses")
plt.plot(list(data), label=stock_name)
plt.legend(loc='upper right')
fig.suptitle(str(year) + "/" + 
             str(month) + " " +  
             stock_name + " alpha" + 
             str(alpha) + "p" + str(period))
plt.show()
fig.savefig("images/" + str(year) + "-" + 
            str(month) + stock_name + 
            "alpha" + str(alpha) + "p" + 
            str(period) + ".png")