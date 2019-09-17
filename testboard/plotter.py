"""Nani."""

import os
import matplotlib.pyplot as plt


class Plotter():
    """docstring for Ploter."""

    @staticmethod
    def acc_box_plot(cls, data, stock, year, features):
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
        # return plt
        filename = "./graphics/{}/{}/{}".format(stock, year, features)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
        plt.close('all')

    @staticmethod
    def loss_epoch_plot(cls, loss_train, stock, year):
        fig, ax_plot = plt.subplots()
        hfont = {'fontname': 'monospace'}
        colors = ['#52D2BC', '#309B8A', '#2460A7', '#21366E']
        ax_plot.set_title("{}-{}".format(stock, year), **hfont)
        ax_plot.set_xlabel('Epoch', **hfont)
        ax_plot.set_ylabel('Loss', **hfont)
        for loss_list, color in zip(loss_train, colors):
            ax_plot.plot(loss_list, color=color)
        plt.show()
        # return fig, ax_plot

    @staticmethod
    def loss_acc_plot(acc, loss, stock, year, features):
        fig1, ax1 = plt.subplots() #figsize=(20, 10)
        hfont = {'fontname': 'monospace'}
        colors = ['#52D2BC', '#309B8A', '#2460A7', '#21366E']
        ax1.set_title("{}-{}".format(stock, year), **hfont)
        ax1.set_xlabel('Epoch', **hfont)
        ax1.set_ylabel('Loss', **hfont)
        for loss_list, color in zip(loss, colors):
            ax1.plot(loss_list, color=color)

        left, bottom, width, height = [0.55, 0.55, 0.3, 0.3]
        ax2 = fig1.add_axes([left, bottom, width, height])
        ax2.set_xlabel('Window Size', **hfont)
        ax2.set_ylabel('Accuracy', **hfont)
        bplot = ax2.boxplot(acc, patch_artist=True, sym='.')
        ax2.set_xticklabels(['25%', '50%', '75%', '100%'])
        colors = ['#52D2BC', '#309B8A', '#2460A7', '#21366E']
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        plt.setp(bplot['medians'], color='#ffffff')

        plt.show()

    # @staticmethod
    # def acc_epoch_plot(acc):
    #     _, ax2 = plt.subplots()
    #     hfont = {'fontname': 'monospace'}
    #     # ax_plot.set_title("{}-{}".format(stock, year), **hfont)
    #     ax_plot.set_xlabel('Epoch', **hfont)
    #     ax_plot.set_ylabel('Loss', **hfont)
    #     ax_plot.plot(acc)
    #     plt.show()
    # #
    # def acc_loss_masterfuckingplot():
    #     fig, (ax1, ax2) = plt.subplots(1, 2)
    #     ax1.plot(x, y)
    #     ax2.plot(x, -y)
        # filename = "./graphics/{}/{}/{}".format(stock, year, features)
        # os.makedirs(os.path.dirname(filename), exist_ok=True)
        # plt.savefig(filename)
        # plt.close('all')
