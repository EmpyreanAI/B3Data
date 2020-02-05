import sys
sys.path.append('../')
sys.path.append('../testboard')
from decimal import Decimal
import matplotlib.pyplot as plt
import numpy as np
from models.denselstm import DenseLSTM
from data_mining.stocks import Stocks
from testboard.data_mining.smote import duplicate_data
from data_mining.stocks import CLOSING, OPENING, MAX_PRICE, MIN_PRICE, MEAN_PRICE, VOLUME
from sklearn.metrics import confusion_matrix

stocks = Stocks(year=2014, cod='VALE3', period=5)
dataset = stocks.selected_fields([CLOSING])
dataset = duplicate_data(dataset)

model = DenseLSTM(input_shape=dataset.shape[1],
                  look_back=6, lstm_cells=150, optimizer='rmsprop')
model.create_data_for_fit(dataset)
model.fit_and_evaluate(batch_size=2, epochs=5000)

model.model.save("VALE3-model.h5")
print("Saved VALE3 model to disk")

acc = model.fit_and_evaluate(5000, 2)['acc']
print(acc)
prediction = model.model.predict(model.test_x)
prediction_labels = [1 if Decimal(i.item()) >= Decimal(0.50) else 0 for i in prediction]
print(prediction_labels)
print(prediction)

begin_test = int(len(dataset) - len(model.test_x))
x = list(range(begin_test, len(dataset)))
y = dataset[begin_test:len(dataset)]
plt.ylabel("Daily Quotation Price")
plt.xlabel("Days")
plt.text(50, 50, "Teste", fontsize=12)
for i in range(len(x)-1):
    if prediction_labels[i] == 1:
        h = plt.scatter(x[i], y[i], marker='^', c='green')
    else:
        l = plt.scatter(x[i], y[i], marker='v', c='red')

plt.legend((h,l), ("Possible higher price in the next day", "Possible lower price in the next day"))

plt.plot(dataset)
plt.show()
