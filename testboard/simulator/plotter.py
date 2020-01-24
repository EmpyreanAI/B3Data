"""Nani."""

import os
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


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
        ax_plot.set_xticklabels([1, 3, 6, 9, 12])
        colors = ['#A0D1CA', '#40C1AC', '#00A3AD', '#007398', '#005A6F']

        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        plt.setp(bplot['medians'], color='#ffffff')
        # return plt
        filename = "./graphics/{}/{}/{}.pdf".format(stock, year, features)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
        plt.close('all')

    @classmethod
    def acc_cells_box_plot(cls, data, stock, year):
        """Nani."""
        _, ax_plot = plt.subplots()
        hfont = {'fontname': 'monospace'}
        ax_plot.set_title("{}-{}".format(stock, year), **hfont)
        ax_plot.set_xlabel('Quantidade de Células LSTM', **hfont)
        ax_plot.set_ylabel('Accuracy', **hfont)
        bplot = ax_plot.boxplot(data, patch_artist=True, sym='.')
        ax_plot.set_xticklabels(['1', '50', '80', '100', '150', '200'])
        colors = ['#A0D1CA', '#40C1AC', '#00A3AD', '#007398', '#005A6F']

        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        plt.setp(bplot['medians'], color='#ffffff')
        # return plt
        filename = "../graphics/{}/{}/acc_cells_box_plot2.png".format(stock,
                                                                      year)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
        plt.close('all')

    @staticmethod
    def loss_epoch_plot(cls, loss_train, stock, year):
        """Nani."""
        fig, ax_plot = plt.subplots()
        hfont = {'fontname': 'monospace'}
        colors = ['#A0D1CA', '#40C1AC', '#00A3AD', '#007398', '#005A6F']
        ax_plot.set_title("{}-{}".format(stock, year), **hfont)
        ax_plot.set_xlabel('Epoch', **hfont)
        ax_plot.set_ylabel('Loss', **hfont)
        for loss_list, color in zip(loss_train, colors):
            ax_plot.plot(loss_list, color=color)
        plt.show()

    @staticmethod
    def loss_acc_plot(acc, loss, stock, year, features):
        """Nani."""
        fig1, ax1 = plt.subplots()  # figsize=(20, 10)
        hfont = {'fontname': 'monospace'}
        colors = ['#A0D1CA', '#40C1AC', '#00A3AD', '#007398', '#005A6F']
        lines = [(0, (1, 10)),':', '-.', '--', '-', '-']
        ax1.set_title("{}-{}".format(stock, year), **hfont)
        ax1.set_xlabel('Epoch', **hfont)
        ax1.set_ylabel('Loss', **hfont)
        for loss_list, color, line in zip(loss, colors, lines):
            ax1.plot(loss_list, color=color, linestyle=line)

        left, bottom, width, height = [0.55, 0.55, 0.3, 0.3]
        ax2 = fig1.add_axes([left, bottom, width, height])
        ax2.set_xlabel('Window Size', **hfont)
        ax2.set_ylabel('Accuracy', **hfont)

        bplot = ax2.boxplot(acc, patch_artist=True, sym='.')
        ax2.set_xticklabels([1, 3, 6, 9, 12])
        for patch, color, line in zip(bplot['boxes'], colors, lines):
            patch.set_linestyle(line)
            patch.set_facecolor(color)

        plt.setp(bplot['medians'], color='#ffffff')

        filename = "./graphics/{}/{}/{}.pdf".format(stock, year, features)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
        plt.close('all')

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, stock, year, features, look_back,
                              normalize=False, cmap=plt.cm.Blues):
        """Print and plot the confusion matrix.

        Normalization can be applied by setting `normalize=True`.
        """
        title = "Confusion Matrix {}-{}".format(stock, year)

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        classes = [1, 0]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # TP / (TP + FN)
        sensitivity = cm[0][0]/(cm[0][0] + cm[0][1])
        sensitivity = round(sensitivity, 4)
        # TP / (TP + FP)
        precision = cm[0][0]/(cm[0][0] + cm[1][0])
        precision = round(precision, 4)
        # TN / (TN + FP)
        specifity = cm[1][1]/(cm[1][1] + cm[1][0])
        specifity = round(specifity, 4)
        # TP + TN / (TP + TN + FP + FN)
        accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1]
                                            + cm[1][0] + cm[1][1])
        accuracy = round(accuracy, 4)

        textstr = "Sensitivity: {}\nPrecision: {}\n".format(sensitivity, precision)
        textstr2 = "Specifity: {}\nAccuracy: {}".format(specifity, accuracy)
        textstr += textstr2

        fig, ax = plt.subplots()
        plt.text(1.05, 0.5, textstr, fontsize=12,
                 verticalalignment='center', transform=ax.transAxes)
        plt.subplots_adjust(right=0.6)

        ax.imshow(cm, interpolation='nearest', cmap=cmap)
        # ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        plt.title(title, fontsize=14)
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # title=title,
               xticklabels=classes, yticklabels=classes,
               ylabel='True label',
               xlabel='Predicted label')

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = np.median(cm)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

        fig.tight_layout()
        filename = "./graphics/{}/{}/boxplot_{}_{}.pdf".format(stock, year,
                                                               look_back,
                                                               features)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
        plt.close('all')
