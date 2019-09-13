import sys
sys.path.append('../')

from testboard.stocks import Stocks
from testboard.stocks import CLOSING, OPENING, MAX_PRICE, MIN_PRICE, MEAN_PRICE, VOLUME
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt


stocks = Stocks(year=2015, cod='PETR3', period=6)
dataset = stocks.selected_fields([CLOSING])

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
ax1.plot(dataset)
ax1.set_xlabel('Número de dados')
ax1.set_ylabel('Valor')

res = []
for i in range(len(dataset)-1):
    res.append(dataset[i])
    value = (dataset[i+1] + dataset[i]) / 2
    res.append(value)

res.append(dataset[-1])
ax2.plot(res)
ax2.set_ylabel('Valor')
ax2.set_xlabel('Número de dados')
plt.tight_layout()
plt.show()



# sm = SMOTE(random_state=42)
# print(len(dataset))
# X_res, Y_res = sm.fit_resample(dataset, range(len(dataset)))
# print(len(dataset))
# print(X_res, Y_res)
