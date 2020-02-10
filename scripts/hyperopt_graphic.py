import pandas
import matplotlib.pyplot as plt
import matplotlib
import numpy
import matplotlib.gridspec as gridspec

space = {
    'batch_size': ["1", "2", "32", "64", "128", "256"],
    'cells': ["1", "50", "80", "100", "150", "200"],
    'optimizers': ['sgd','adam','rmsprop'],
    'look_back': ["1", "3", "6", "9", "12"],
}

df = pandas.read_csv('../results/hyperopt_100_PETR3.csv')

# import pdb; pdb.set_trace()
cells = [space['cells'][int(df.iloc[i]['cells'][1])] for i in range(len(df[['cells']]))]
batch_size = [space['batch_size'][int(df.iloc[i]['batch_size'][1])] for i in range(len(df[['batch_size']]))]
look_back = [space['look_back'][int(df.iloc[i]['look_back'][1])] for i in range(len(df[['look_back']]))]
optimizers = [space['optimizers'][int(df.iloc[i]['optimizers'][1])] for i in range(len(df[['optimizers']]))]
loss = [(df.iloc[i]['loss'])*-1 for i in range(len(df[['optimizers']]))]


ax = plt.subplot(2, 4, 1)
plt.suptitle("PETR3", fontsize=16)
ax.set_title('LSTM Units')
bins = numpy.arange(7) - 0.5
plt.hist(cells, bins=bins, ec='black')
plt.ylabel('Selected Times')

ax = plt.subplot(2, 4, 2)
ax.set_title('Batch Size')
bins = numpy.arange(7) - 0.5
plt.hist(batch_size, bins=bins, ec='black')
plt.ylabel('Selected Times')

ax = plt.subplot(2, 4, 3)
ax.set_title('Time-steps')
bins = numpy.arange(6) - 0.5
plt.hist(look_back, bins=bins, ec='black')
plt.ylabel('Selected Times')

ax = plt.subplot(2, 4, 4)
ax.set_title('Optimizers')
bins = numpy.arange(4) - 0.5
plt.hist(optimizers, bins=bins, ec='black')
plt.ylabel('Selected Times')

avg = numpy.polyfit(df[['index']].squeeze().tolist(), loss, 1)
poly = numpy.poly1d(avg)

ax = plt.subplot(2, 1, 2)
ax.set_title('F1 Score')
plt.plot(poly(df[['index']].squeeze().tolist()), zorder=10)
ax = plt.subplot(2, 1, 2)
plt.plot(df[['index']], loss)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
