import numpy as np
from matplotlib import pyplot as plt


def display_grey_img_list(imgs, ncols, figsize=None, img_names=None):
    img_nbr = len(imgs)
    nrows = int(img_nbr / ncols)
    if img_nbr % ncols != 0:
        nrows += 1
    if figsize is None:
        figsize = (10, 5 * nrows)
    else:
        figsize = (figsize[0], figsize[1] * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1 or ncols == 1:
        for i, img in enumerate(imgs):
            axes[i].imshow(img, cmap='Greys_r')
            axes[i].axis('off')
            if img_names is not None:
                axes[i].set_title(img_names[i])
    else:
        for y in range(nrows):
            for x in range(ncols):
                img_index = x + y * ncols
                if img_index < img_nbr:
                    axes[y, x].imshow(imgs[img_index], cmap='Greys_r')
                    if img_names is not None:
                        axes[y, x].set_title(img_names[img_index])
                axes[y, x].axis('off')
    plt.tight_layout()
    plt.show()


def display_img_and_hist(gray_images, figure_size):
    num_pairs = len(gray_images)
    fig, axes = plt.subplots(num_pairs, 2, figsize=figure_size)

    for i, img in enumerate(gray_images):
        axes[i, 0].imshow(img, cmap='Greys_r')
        axes[i, 0].axis('off')

        histogram, group_edges = np.histogram(img, bins=128)
        group_centers = []
        for k in range(len(group_edges) - 1):
            group_centers.append((group_edges[k] + group_edges[k + 1]) / 2.)
        axes[i, 1].bar(group_centers, histogram)
    plt.tight_layout()
    plt.show()
