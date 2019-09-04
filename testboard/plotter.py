import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


class Plotter(object):
    """docstring for Ploter."""

    def box_plot(self, data, stock, year, features):
        fig, ax = plt.subplots()
        hfont = {'fontname': 'monospace'}
        ax.set_title("{}-{}".format(stock, year), **hfont)
        ax.set_xlabel('Window Size', **hfont)
        ax.set_ylabel('Accuracy', **hfont)
        bplot = ax.boxplot(data, patch_artist=True, sym='.')
        ax.set_xticklabels(['25%', '50%', '75%', '100%'])
        colors = ['#52D2BC', '#309B8A', '#2460A7', '#21366E']
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        plt.setp(bplot['medians'], color='#ffffff')
        filename = "../graphics/{}/{}/something.png".format(stock, year)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
