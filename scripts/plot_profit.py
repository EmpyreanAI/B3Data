import sys
sys.path.append('../')
sys.path.append('../testboard')
from decimal import Decimal
import matplotlib.pyplot as plt
import numpy as np
from models.denselstm import DenseLSTM
from data_mining.stocks import Stocks
from data_mining.stocks import CLOSING, OPENING, MAX_PRICE, MIN_PRICE, MEAN_PRICE, VOLUME

wallet = 1000
stock_wallet = 0

def buy(value):
    global wallet
    global stock_wallet
    wallet = wallet - value
    stock_wallet += 1

def sell(value):
    global wallet
    global stock_wallet
    wallet = wallet + value
    stock_wallet -= 1

stocks = Stocks(year=2014, cod='VALE3', period=6)
dataset = stocks.selected_fields([CLOSING])

look_back = 0.25

new_look_back = (len(dataset)*0.3)*look_back
model = DenseLSTM(input_shape=dataset.shape[1],
                  look_back=int(new_look_back), lstm_cells=150)
model.create_data_for_fit(dataset)
model.fit_and_evaluate(epochs=5000)

prediction = model.model.predict(model.test_x)
prediction_labels = [1 if Decimal(i.item()) >= Decimal(0.50) else 0 for i in prediction]
print(prediction_labels)
print(prediction)

begin_test = int(len(dataset) - len(model.test_x))
x = list(range(begin_test, len(dataset)))
y = dataset[begin_test:len(dataset)]

for i in range(len(x)-1):
    if prediction_labels[i] == 1:
        plt.scatter(x[i], y[i], marker='^', c='green')
    else:
        plt.scatter(x[i], y[i], marker='v', c='red')

    if prediction_labels[i+1] == 1:
        buy(y[i])
    else:
        sell(y[i])

if stock_wallet > 0:
    for i in range(stock_wallet):
        sell(y[-1])

plt.text(0.01, 0.1, 'wallet = {}'.format(*wallet), transform=plt.gca().transAxes)
print(wallet)
plt.plot(dataset)
plt.show()
