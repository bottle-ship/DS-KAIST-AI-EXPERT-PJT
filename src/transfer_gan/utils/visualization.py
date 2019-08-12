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
