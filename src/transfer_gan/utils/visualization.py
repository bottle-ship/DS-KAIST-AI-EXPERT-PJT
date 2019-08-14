import matplotlib.pyplot as plt
import numpy as np


def show_image_from_dataset(x, y, class_names=None):
    selected_idx = np.random.randint(0, x.shape[0], 10)

    fig = plt.figure(figsize=(26, 18))
    ax = list()

    for i in range(0, 10):
        ax.append(fig.add_subplot(2, 5, i + 1))
        ax[i].imshow(x[selected_idx[i]])
        label = y.ravel()[selected_idx[i]]
        if class_names is not None:
            ax[i].set_title("%d; %s" % (label, class_names[label]))
        else:
            ax[i].set_title("%d" % label)
        ax[i].axis("off")

    plt.subplots_adjust(wspace=0.1, hspace=0.0)
    plt.show()
    plt.close()


def show_generated_image(x, n_col=8, filename=None):
    total_image = x.shape[0]
    img_channel = x.shape[-1]
    n_row = int(np.ceil(x.shape[0] / n_col))

    fig = plt.figure(figsize=(5 * (n_col / 8), 5 * (n_row / 8)))
    fig.set_facecolor('k')
    ax = list()

    for i in range(0, total_image):
        ax.append(fig.add_subplot(n_row, n_col, i + 1))
        if img_channel == 1:
            ax[i].imshow(x[i, :, :, 0], cmap='gray')
        else:
            ax[i].imshow(x[i])
        ax[i].axis("off")

    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()
