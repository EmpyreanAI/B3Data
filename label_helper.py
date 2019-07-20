import math
import numpy
import indicators

class LabelHelper(object):

    @staticmethod
    def up_down_with_interval(dataset, look_back=2):
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back):
            day_t = dataset[i:(i+look_back)]
            day_t1 = dataset[i + look_back]
            dataX.append(day_t)
            if day_t.mean() > day_t1:
                dataY.append(0)
            else:
                dataY.append(1)

        return numpy.array(dataX), numpy.array(dataY)