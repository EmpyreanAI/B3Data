import numpy

def duplicate_data(dataset):
    res = []
    for i in range(len(dataset)-1):
        res.append(dataset[i])
        value = (dataset[i+1] + dataset[i]) / 2
        res.append(value)
    res = numpy.array(res)
    return res
