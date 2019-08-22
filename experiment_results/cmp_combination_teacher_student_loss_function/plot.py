from transfer_gan.utils.visualization import show_compare_fid_history_n_bar_plot


if __name__ == '__main__':
    history_path_list = ['history_dc_dc.csv',
                         'history_dc_wgangp.csv'
                         ]
    label_list = ['T: DCGAN | S: DCGAN',
                  'T: DCGAN | S: WGAN-GP']

    title = "Initialize by teacher weights @ 10000 iteration\n" \
            "Teacher: DCGAN / Tiny imagenet # 15000 → Student: DCGAN or WGAN-GP / Cifar10 # 1000"

    show_compare_fid_history_n_bar_plot(
        history_path_list, label_list, bar_xlim=10000, xlim=10000, title=title, filename='compare.png'
    )
