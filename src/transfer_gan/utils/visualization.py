import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def show_image_from_dataset(x, y, class_names=None):
    img_channel = x.shape[-1]

    if x.dtype != 'uint8':
        x = x.astype('uint8')

    selected_idx = np.random.randint(0, x.shape[0], 10)

    fig = plt.figure(figsize=(26, 18))
    ax = list()

    for i in range(0, 10):
        ax.append(fig.add_subplot(2, 5, i + 1))
        if img_channel == 1:
            ax[i].imshow(x[selected_idx[i], :, :, 0], cmap='gray')
        else:
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

    if x.dtype != 'uint8':
        x = x.astype('uint8')

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


def show_compare_fid_history(history_path_list, label_list, xlim=None, title=None, filename=None):
    df_list = list()
    for history_path in history_path_list:
        df_list.append(pd.read_csv(history_path))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    max_epochs = 0

    for df, label in zip(df_list, label_list):
        epochs = df['Epochs'].values.max()
        if epochs > max_epochs:
            max_epochs = epochs

        ax.plot(df['Epochs'].values.tolist(), df['FID'].values.tolist(), label=label)

    ax.set_xlabel('Iteration', fontsize=24)
    ax.set_ylabel('FID', fontsize=24)

    if xlim is None:
        ax.set_xlim(0, max_epochs)
    else:
        ax.set_xlim(0, xlim)

    plt.legend()

    if title is not None:
        plt.title(title, fontsize=24)

    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()


def show_compare_fid_bar_plot(history_path_list, label_list, max_epochs, title=None, filename=None):
    df_list = list()
    for history_path in history_path_list:
        df_list.append(pd.read_csv(history_path))

    for i in range(0, len(df_list)):
        df_list[i] = df_list[i][df_list[i]['Epochs'] <= max_epochs]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    idx = 0
    for df, label in zip(df_list, label_list):
        ax.bar(idx, df['FID'].min(), label=label)
        idx += 1

    ax.set_ylabel('FID', fontsize=24)
    ax.set_xticks([])

    plt.legend()

    if title is not None:
        plt.title(title, fontsize=24)

    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()


def show_compare_fid_history_n_bar_plot(history_path_list, label_list, bar_xlim, xlim=None, title=None, filename=None):
    df_list = list()
    for history_path in history_path_list:
        df_list.append(pd.read_csv(history_path))

    fig = plt.figure(figsize=(15, 6))

    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    max_epochs = 0

    for df, label in zip(df_list, label_list):
        epochs = df['Epochs'].values.max()
        if epochs > max_epochs:
            max_epochs = epochs

        ax1.plot(df['Epochs'].values.tolist(), df['FID'].values.tolist(), label=label)

    ax1.set_xlabel('Iteration', fontsize=16)
    ax1.set_ylabel('FID', fontsize=16)

    for i in range(0, len(df_list)):
        df_list[i] = df_list[i][df_list[i]['Epochs'] <= bar_xlim]

    idx = 0
    fid_list = list()
    for df, label in zip(df_list, label_list):
        value = df['FID'].min()
        fid_list.append('%.2f' % value)
        ax2.bar(idx, value, label=label)
        idx += 1

    rects = ax2.patches

    for rect, label in zip(rects, fid_list):
        height = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width() / 2, height, label, ha='center', va='bottom', fontweight='bold')

    ax2.set_ylabel('FID', fontsize=16)
    ax2.set_xticks([])

    if xlim is None:
        ax1.set_xlim(0, max_epochs)
    else:
        ax1.set_xlim(0, xlim)

    if title is not None:
        plt.title(title, fontsize=18, position=(-0.1, 1))

    ax1.legend()
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.15, hspace=0.0)

    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()
