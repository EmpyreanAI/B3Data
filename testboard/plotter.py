import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


class Ploter(object):
    """docstring for Ploter."""

    def box_plot(self, data):
        fig, ax = plt.subplots()
        # csfont = {'fontname':'Comic Sans MS'}
        hfont = {'fontname': 'monospace'}
        # ax.set_title('PLS KILL ME', **hfont)
        ax.set_xlabel('Window Size', **hfont)
        ax.set_ylabel('Accuracy', **hfont)
        data = self.gen_random_data()

        bplot = ax.boxplot(data, patch_artist=True, sym='.')
        ax.set_xticklabels(['25%', '50%', '75%', '100%'])
        colors = ['#52D2BC', '#309B8A', '#2460A7', '#21366E']
        print(bplot)

        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

        plt.setp(bplot['medians'], color='#ffffff')

        plt.show()

    def gen_random_data(self):
        # Fixing random state for reproducibility
        np.random.seed(19680801)

        # fake up some data
        spread = np.random.rand(50) * 100
        center = np.ones(25) * 50
        flier_high = np.random.rand(10) * 100 + 100
        flier_low = np.random.rand(10) * -100
        data = np.concatenate((spread, center, flier_high, flier_low))
        spread = np.random.rand(50) * 100
        center = np.ones(25) * 40
        flier_high = np.random.rand(10) * 100 + 100
        flier_low = np.random.rand(10) * -100
        d2 = np.concatenate((spread, center, flier_high, flier_low))
        data.shape = (-1, 1)
        d2.shape = (-1, 1)
        # Making a 2-D array only works if all the columns are the
        # same length.  If they are not, then use a list instead.
        # This is actually more efficient because boxplot converts
        # a 2-D array into a list of vectors internally anyway.
        data = [data, d2, d2[::2, 0], data]
        print(data)
        return data


plot = Ploter()
plot.box_plot([])
