"""Nani."""

import os
import matplotlib.pyplot as plt


class Plotter():
    """docstring for Ploter."""

    @staticmethod
    def box_plot(data, stock, year, features):
        """Nani."""
        _, ax_plot = plt.subplots()
        hfont = {'fontname': 'monospace'}
        ax_plot.set_title("{}-{}".format(stock, year), **hfont)
        ax_plot.set_xlabel('Window Size', **hfont)
        ax_plot.set_ylabel('Accuracy', **hfont)
        bplot = ax_plot.boxplot(data, patch_artist=True, sym='.')
        ax_plot.set_xticklabels(['25%', '50%', '75%', '100%'])
        colors = ['#52D2BC', '#309B8A', '#2460A7', '#21366E']
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        plt.setp(bplot['medians'], color='#ffffff')
        filename = "./graphics/{}/{}/{}".format(stock, year, features)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
        plt.close('all')

    @staticmethod
    def loss_epoch_plot(loss_train):
        _, ax_plot = plt.subplots()
        hfont = {'fontname': 'monospace'}
        # ax_plot.set_title("{}-{}".format(stock, year), **hfont)
        ax_plot.set_xlabel('Epoch', **hfont)
        ax_plot.set_ylabel('Loss', **hfont)
        ax_plot.plot(loss_train)
        plt.show()

    @staticmethod
    def acc_epoch_plot(acc):
        _, ax_plot = plt.subplots()
        hfont = {'fontname': 'monospace'}
        # ax_plot.set_title("{}-{}".format(stock, year), **hfont)
        ax_plot.set_xlabel('Epoch', **hfont)
        ax_plot.set_ylabel('Loss', **hfont)
        ax_plot.plot(acc)
        plt.show()
    #
    # def acc_loss_masterfuckingplot():
    #     fig, (ax1, ax2) = plt.subplots(1, 2)
    #     ax1.plot(x, y)
    #     ax2.plot(x, -y)
        # filename = "./graphics/{}/{}/{}".format(stock, year, features)
        # os.makedirs(os.path.dirname(filename), exist_ok=True)
        # plt.savefig(filename)
        # plt.close('all')
