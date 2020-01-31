import matplotlib.pyplot as plt
# import Orange, orngStat
import pandas
import numpy

df = pandas.read_csv('../results/cells_experiment_ABEV.csv', index_col=0)

initial_rank = 1
names = ['1 unit', '50 units', '80 units', '100 units', '150 units', '200 units']
table_ranks = []

for index, row in df.iterrows():
    initial_rank = 1
    rank_row_dict = {}
    sorted = row.sort_values(ascending=False)
    for i in sorted.index.tolist():
        rank_row_dict.update({i: initial_rank})
        initial_rank += 1

    rank_row_list = []
    for name in names:
        rank_row_list.append(rank_row_dict[name])

    table_ranks.append(rank_row_list)

print(table_ranks)
avranks = numpy.mean(table_ranks, axis=0)
print(avranks)
