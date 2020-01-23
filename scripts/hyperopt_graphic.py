import pandas
import matplotlib.pyplot as plt
import matplotlib
import numpy

space = {
    'batch_size': ["1", "2", "32", "64", "128", "256"],
    'cells': ["1", "50", "80", "100", "150", "200"],
    'optimizers': ['sgd','adam','rmsprop'],
    'look_back': ["1", "3", "6", "9", "12"],
}

df = pandas.read_csv('../results/hyperopt_100_ABEV3.csv')

# import pdb; pdb.set_trace()
cells = [space['cells'][int(df.iloc[i]['cells'][1])] for i in range(len(df[['cells']]))]
batch_size = [space['batch_size'][int(df.iloc[i]['batch_size'][1])] for i in range(len(df[['batch_size']]))]
look_back = [space['look_back'][int(df.iloc[i]['look_back'][1])] for i in range(len(df[['look_back']]))]
optimizers = [space['optimizers'][int(df.iloc[i]['optimizers'][1])] for i in range(len(df[['optimizers']]))]
loss = [df.iloc[i]['loss'] for i in range(len(df[['optimizers']]))]


ax = plt.subplot(2, 4, 1)
ax.set_title('Cells')
bins = numpy.arange(7) - 0.5
plt.hist(cells, bins=bins, ec='black')

ax = plt.subplot(2, 4, 2)
ax.set_title('batch_size')
bins = numpy.arange(7) - 0.5
plt.hist(batch_size, bins=bins, ec='black')

ax = plt.subplot(2, 4, 3)
ax.set_title('timesteps')
bins = numpy.arange(6) - 0.5
plt.hist(look_back, bins=bins, ec='black')
ax = plt.subplot(2, 4, 4)

ax.set_title('Optmizers')
bins = numpy.arange(4) - 0.5
plt.hist(optimizers, bins=bins, ec='black')

avg = numpy.polyfit(df[['index']].squeeze().tolist(), df[['loss']].squeeze().tolist(), 1)
poly = numpy.poly1d(avg)

ax = plt.subplot(2, 1, 2)
ax.set_title('F1 Score')
plt.plot(poly(df[['index']].squeeze().tolist()), zorder=10)
ax = plt.subplot(2, 1, 2)
plt.plot(df[['index']], df[['loss']])

plt.tight_layout()
plt.show()
