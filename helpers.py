import math
import numpy
import indicators

class Helpers(object):
    @staticmethod
    def create_dataset(dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-1):
            day_t = dataset[i:(i+look_back), 0]
            day_t1 = dataset[i + look_back, 0]
            dataX.append(day_t)
            print(day_t)
            if day_t > day_t1:
                dataY.append(0)
            else:
                dataY.append(1)

        # import pdb; pdb.set_trace()
        return numpy.array(dataX), numpy.array(dataY)

    @staticmethod
    def to_zero_one(vector):
        for i in range(0, len(vector)-1):
            if vector[i] > vector[i+1]:
                vector[i] = 0
            else:
                vector[i] = 1

        return vector

    @staticmethod
    def regenerate_dataset(dataframe, indicator_method=None, window_size=3):
        dataset = dataframe.values
        dataset = dataset.astype('float32')

        if indicator_method is "price":
            dataset = numpy.asarray(dataset)
        else:
            dataset = numpy.asarray(getattr(indicators, indicator_method)(dataset, window_size))
        dataset = dataset.reshape(len(dataset), 1)
        return dataset