from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def display_loss_val(cols: list[str], losses: list[float], val_losses: list[float], width=0.35):
    x = np.arange(len(cols))

    fig, ax = plt.subplots()
    ax.bar(x - width / 2, losses, width)
    ax.bar(x + width / 2, val_losses, width)
    ax.xaxis.set_ticks(x)
    ax.set_xticklabels(cols)

    plt.xlabel('Models')
    plt.ylabel('Losses and val_losses')
    plt.title('Bar Plot of errors')
    plt.show()


def display_errors(cols: list[str], errors: list[float]):
    errors_index = range(len(errors))
    width = 0.5

    plt.bar(errors_index, errors, width)
    plt.xticks(errors_index, cols)

    plt.xlabel('Models')
    plt.ylabel('Errors')
    plt.title('Bar Plot of errors')
    plt.show()


def trainings_errors_boxplot(errors_list: list[pd.DataFrame], fig_size, cols: list[str], ylim: Optional[tuple[int, int]] = None):
    formatted_errors = []
    for errors in errors_list:
        formatted_errors.append(errors[0].values)
    plt.figure(figsize=fig_size)
    plt.boxplot(formatted_errors, labels=cols)
    plt.grid()
    plt.title('Boxplot for all solutions errors')
    plt.xlabel('Solutions')
    plt.ylabel('Errors')
    plt.tight_layout()
    if ylim is not None:
        plt.ylim(ylim)
    plt.show()


def trainings_errors_scatterplot(errors: list[list], fig_size, cols: list[str], ylim: Optional[tuple[int, int]] = None):
    plt.figure(figsize=fig_size)
    x_positions = range(len(cols)+1)
    for i, col in enumerate(cols):
        plt.scatter([i+1] * len(errors[i]), errors[i], label=col)

    plt.xticks(x_positions, [''] + cols)

    for pos in x_positions:
        plt.axvline(x=pos, color='gray', linestyle='dotted')

    plt.xlabel('Solutions')
    plt.ylabel('Errors')
    plt.legend()
    plt.xlim((0.5, len(cols) + 0.5))
    plt.tight_layout()
    if ylim is not None:
        plt.ylim(ylim)
    plt.show()


def plot_metrics(history, loss_chart_name, metric_names: list[str], ylim: Optional[tuple[int, int]] = None):
    plt.figure(figsize=(5, 5))
    for metric_name in metric_names:
        metric = history[metric_name]
        plt.plot(range(len(metric)), metric, label=metric_name)
    plt.legend()
    plt.grid()
    if ylim is not None:
        plt.ylim(ylim)
    plt.xlabel("# of epochs")
    plt.ylabel(loss_chart_name)
    plt.show()
