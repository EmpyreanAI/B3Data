import pandas
import matplotlib.pyplot as plt

df = pandas.read_csv('../results/hyperopt_33max_val.csv')

plt.plot(df[['index']], df[['loss']])
plt.show()