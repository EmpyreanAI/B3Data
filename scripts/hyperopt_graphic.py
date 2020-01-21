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

df = pandas.read_csv('../results/hyperopt_100_VALE3.csv')

cells = [space['cells'][int(df.iloc[i]['cells'][1])] for i in range(len(df[['cells']]))]
batch_size = [space['batch_size'][int(df.iloc[i]['batch_size'][1])] for i in range(len(df[['batch_size']]))]
look_back = [space['look_back'][int(df.iloc[i]['look_back'][1])] for i in range(len(df[['look_back']]))]
optimizers = [space['optimizers'][int(df.iloc[i]['optimizers'][1])] for i in range(len(df[['optimizers']]))]
loss = [df.iloc[i]['loss'] for i in range(len(df[['optimizers']]))]


ax = plt.subplot(2, 4, 1)
ax.set_title('Cells')
# plt.yticks(["1", "50", "80", "100", "150", "200"])
plt.plot(df[['index']], cells, 'bo')

ax = plt.subplot(2, 4, 2)
ax.set_title('batch_size')
# plt.axis('normal')
# plt.yticks([1, 2, 32, 64, 128, 256])
plt.plot(df[['index']], batch_size, 'go')

ax = plt.subplot(2, 4, 3)
ax.set_title('timesteps')
# plt.axis('normal')
plt.plot(df[['index']], look_back, 'ro')
#
ax = plt.subplot(2, 4, 4)
ax.set_title('Optmizers')
plt.plot(df[['index']], optimizers, 'yo')

avg = numpy.polyfit(df[['index']].squeeze().tolist(), df[['loss']].squeeze().tolist(), 1)
poly = numpy.poly1d(avg)

ax = plt.subplot(2, 1, 2)
ax.set_title('F1 Score')
plt.plot(poly(df[['index']].squeeze().tolist()), zorder=10)
ax = plt.subplot(2, 1, 2)
plt.plot(df[['index']], df[['loss']])

plt.tight_layout()
plt.show()
