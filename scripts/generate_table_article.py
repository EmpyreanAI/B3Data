import matplotlib.pyplot as plt
# import Orange, orngStat
import pandas
import numpy
import sys
import os

df = pandas.read_csv('../results/posthoc_PETR3.csv', index_col=0)

initial_rank = 1
names = ['cells_1', 'cells_50', 'cells_80', 'cells_100', 'cells_150', 'cells_200']
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

final_df_names = ['1 unit', '50 units', '80 units', '100 units', '150 units', '200 units']
final_df = pandas.DataFrame(columns=final_df_names)

pandas.set_option("display.precision", 5)

for index, row in df.iterrows():
    final_df.set_value(index+1, '1 unit', str(round(row['cells_1'], 5)) + " (" + str(table_ranks[index][0]) + ")")
    final_df.set_value(index+1, '50 units', str(round(row['cells_50'], 5)) + " (" + str(table_ranks[index][1]) + ")")
    final_df.set_value(index+1, '80 units', str(round(row['cells_80'], 5)) + " (" + str(table_ranks[index][2]) + ")")
    final_df.set_value(index+1, '100 units', str(round(row['cells_100'], 5)) + " (" + str(table_ranks[index][3]) + ")")
    final_df.set_value(index+1, '150 units', str(round(row['cells_150'], 5)) + " (" + str(table_ranks[index][4]) + ")")
    final_df.set_value(index+1, '200 units', str(round(row['cells_200'], 5)) + " (" + str(table_ranks[index][5]) + ")")

final_df = final_df.append({'1 unit': avranks[0],
                            '50 units': avranks[1],
                            '80 units': avranks[2],
                            '100 units': avranks[3],
                            '150 units': avranks[4],
                            '200 units': avranks[5]}, ignore_index=True)
final_df.index += 1

outname = 'posthoc_rank_table_PETR3.csv'
outdir = '../results/rank_tables'
if not os.path.exists(outdir):
    os.mkdir(outdir)

fullname = os.path.join(outdir, outname)

final_df.to_csv(fullname, mode='a')
print(final_df)
