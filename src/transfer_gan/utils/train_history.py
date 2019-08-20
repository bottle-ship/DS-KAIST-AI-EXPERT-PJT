import os

import pandas as pd


def get_train_history(target_path):
    filename_list = os.listdir(target_path)

    data = list()
    for filename in filename_list:
        if filename.endswith('png') and 'fid' in filename and 'iteration' in filename:
            filename = filename.replace('.png', '')

            filename = filename.split('_')

            iteration = int(filename[1])
            fid = float(filename[-1])

            data.append([iteration, fid])

    df_history = pd.DataFrame(data, columns=['Epochs', 'FID'])
    df_history = df_history.sort_values(by=['Epochs']).reset_index(drop=True)

    df_history.to_csv('history.csv', index=False)
