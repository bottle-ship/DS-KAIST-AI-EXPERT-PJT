# import os
#
# from transfer_gan.utils.visualization import show_compare_fid_history
# from transfer_gan.utils.visualization import show_compare_fid_bar_plot
#
#
# ROOT_PATH = '/home/com12/PycharmProjects/DS-KAIST-AI-EXPERT-PJT'
# SRC_MODEL_ROOT_PATH = '/home/com12/PycharmProjects/DS-KAIST-AI-EXPERT-PJT/source_model'
#
#
# if __name__ == '__main__':
#     history_path_list = [os.path.join(ROOT_PATH, 'source_model', 'student_dcgan_cifar10_full_baseline', 'history.csv'),
#                          os.path.join(SRC_MODEL_ROOT_PATH, 'student_dcgan_cifar10_subset_baseline', 'history.csv'),
#                          os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'init_teacher_weight', 'history.csv'),
#                          'history_d.csv',
#                          'history_g.csv',
#                          'history_gd.csv'
#                          ]
#     label_list = ['Baseline (DCGAN  / Cifar10 # 50000)',
#                   'Baseline (DCGAN  / Cifar10 # 1000)',
#                   'All layer initialize by teacher weights',
#                   'Discriminator',
#                   'Generator',
#                   'Discriminator and Generator'
#                   ]
#
#     title = "Last layer random initialize @ 10000 iteration\n" \
#             "Teacher: DCGAN / Tiny imagenet # 15000 → Student: DCGAN / Cifar10 # 1000"
#
#     show_compare_fid_history(history_path_list, label_list, xlim=10000, title=title)
#
#     show_compare_fid_bar_plot(history_path_list, label_list, max_epochs=10000, title=title)
from transfer_gan.utils.visualization import show_compare_fid_history_n_bar_plot


if __name__ == '__main__':
    history_path_list = ['history_d.csv',
                         'history_g.csv',
                         'history_gd.csv'
                         ]
    label_list = ['Discriminator',
                  'Generator',
                  'Both Discriminator and Generator']

    title = "The effect of weights initialize\n" \
            "Teacher: DCGAN / Tiny imagenet # 15000 → Student: DCGAN or WGAN-GP / Cifar10 # 1000"

    show_compare_fid_history_n_bar_plot(
        history_path_list, label_list, bar_xlim=10000, xlim=10000, title=title, filename='compare.png'
    )
