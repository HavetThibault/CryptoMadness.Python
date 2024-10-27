from matplotlib import pyplot as plt

from model.training.analyse.dataset_cartographer import sort_by_piece_name, get_pieces_stats, sort_pieces_stats_by_mean, \
    sort_pieces_stats_by_std
from plot.common import display_bar_plot


def display_cartography(cartography: dict[str, tuple[float, float]], title: str):
    for mean_std in cartography.values():
        plt.scatter(mean_std[1], mean_std[0], color='#ff7f0e')
    plt.grid()
    plt.xlabel('Standard deviation')
    plt.ylabel('Mean')
    plt.title(title)
    plt.show()


def display_cartography_stats(train_cartography, training_mem_name: str):
    display_cartography(train_cartography, training_mem_name)
    sorted_train = sort_by_piece_name(train_cartography)
    pieces_stats = get_pieces_stats(sorted_train)

    pieces_stats.sort(key=sort_pieces_stats_by_mean)
    mean_piece_names = []
    errors_means = []
    for piece_name, errors_mean, errors_std in pieces_stats:
        mean_piece_names.append(piece_name)
        errors_means.append(errors_mean)
    display_bar_plot(mean_piece_names, errors_means, (10, 5), '', 'Error means')

    pieces_stats.sort(key=sort_pieces_stats_by_std)
    mean_piece_names = []
    errors_stds = []
    for piece_name, errors_mean, errors_std in pieces_stats:
        mean_piece_names.append(piece_name)
        errors_stds.append(errors_std)
    display_bar_plot(mean_piece_names, errors_stds, (10, 5), '', 'Error stds')
