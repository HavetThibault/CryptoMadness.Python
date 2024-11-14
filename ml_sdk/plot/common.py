import numpy as np
from matplotlib import pyplot as plt


def display_bar_plot(labels: list[str], values: list, figsize: tuple[int, int], ylabel, title, bar_width=0.35):
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x, values, bar_width, label='Error mean')
    ax.xaxis.set_ticks(x)
    ax.set_xticklabels(labels)

    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def display_line_plot(x, y, figsize: tuple[int, int], title, xlabel=None, ylabel=None, ylim=None, xlim=None):
    plt.figure(figsize=figsize)
    plt.plot(x, y)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.title(title)
    plt.show()
