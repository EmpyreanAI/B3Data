import pandas
import matplotlib.pyplot as plt
import matplotlib

space = {
    'batch_size': [1, 2, 32, 64, 128, 256],
    'cells': [1, 50, 80, 100, 150, 200],
    'optimizers': ['sgd','adam','rmsprop'],
    'look_back': [1, 3, 6, 9, 12],
}

df = pandas.read_csv('../results/hyperopt_100_VALE3.csv')

# plt.plot(df[['index']], df[['loss']])

# import pdb; pdb.set_trace()
cells = [int(space['cells'][int(df.iloc[i]['cells'][1])]) for i in range(len(df[['cells']]))]
batch_size = [int(space['batch_size'][int(df.iloc[i]['batch_size'][1])]) for i in range(len(df[['batch_size']]))]
look_back = [int(space['look_back'][int(df.iloc[i]['look_back'][1])]) for i in range(len(df[['look_back']]))]
optimizers = [space['optimizers'][int(df.iloc[i]['optimizers'][1])] for i in range(len(df[['optimizers']]))]

# plt.plot(df[['index']], df[['loss']])

ax = plt.subplot(1, 4, 1)
ax.set_title('Cells')
plt.yticks([1, 50, 80, 100, 150, 200])
plt.plot(df[['index']], cells, 'bo')
ax = plt.subplot(1, 4, 2)
ax.set_title('batch_size')
plt.axis('normal')
plt.yticks([1, 2, 32, 64, 128, 256])
plt.plot(df[['index']], batch_size, 'go')
ax = plt.subplot(1, 4, 3)
ax.set_title('timesteps')
plt.axis('normal')
plt.yticks([1, 3, 6, 9, 12])
plt.plot(df[['index']], look_back, 'ro')

ax = plt.subplot(1, 4, 4)
ax.set_title('Optmizers')
# plt.yticks(['sgd', 'adam', 'rmsprop'])
plt.plot(df[['index']], optimizers, 'yo')

plt.tight_layout()
plt.show()
