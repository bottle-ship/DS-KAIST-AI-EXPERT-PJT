import os

from transfer_gan.utils.visualization import show_compare_fid_history
from transfer_gan.utils.visualization import show_compare_fid_bar_plot


ROOT_PATH = '/home/com12/PycharmProjects/DS-KAIST-AI-EXPERT-PJT'
SRC_MODEL_ROOT_PATH = '/home/com12/PycharmProjects/DS-KAIST-AI-EXPERT-PJT/source_model'


if __name__ == '__main__':
    history_path_list = [os.path.join(ROOT_PATH, 'source_model', 'student_dcgan_cifar10_full_baseline', 'history.csv'),
                         os.path.join(SRC_MODEL_ROOT_PATH, 'student_dcgan_cifar10_subset_baseline', 'history.csv'),
                         'history.csv',
                         'history_full.csv'
                         ]
    label_list = ['Baseline (DCGAN  / Cifar10 # 50000)',
                  'Baseline (DCGAN  / Cifar10 # 1000)',
                  'Tiny imagenet subset',
                  'Tiny imagenet fullset'
                  ]

    title = "Initialize by teacher's weight @ 10000 iteration"

    show_compare_fid_history(history_path_list, label_list, xlim=10000, title=title)

    show_compare_fid_bar_plot(history_path_list, label_list, max_epochs=10000, title=title)
