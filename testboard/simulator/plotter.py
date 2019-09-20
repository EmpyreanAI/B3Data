"""Nani."""

import os
import numpy as np

import matplotlib.pyplot as plt
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


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
        """Nani."""
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
        """Nani."""
        fig1, ax1 = plt.subplots()  # figsize=(20, 10)
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

    @classmethod
    def gen_random_data(cls):
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        class_names = iris.target_names

        # Split the data into a training set and a test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

        # Run classifier, using a model that is too regularized (C too low) to see
        # the impact on the results
        classifier = svm.SVC(kernel='linear', C=0.01)
        y_pred = classifier.fit(X_train, y_train).predict(X_test)
        cls.plot_confusion_matrix(y_test, y_pred, classes=class_names,
                              title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
        cls.plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,
                              title='Normalized confusion matrix')

        plt.show()

    def plot_confusion_matrix(y_true, y_pred, classes,
                              normalize=False,
                              title=None,
                              cmap=plt.cm.Blues):
        """Print and plot the confusion matrix.

        Normalization can be applied by setting `normalize=True`.
        """
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        classes = classes[unique_labels(y_true, y_pred)]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        return ax

if __name__ == '__main__':
    plot = Plotter()
    plot.gen_random_data()
