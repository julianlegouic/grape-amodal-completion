"""
    Author: Julian LE GOUIC
    Python version: 3.6

    IMP Lab - Osaka Prefecture University.
"""

import argparse
import dask.dataframe as dd
import numpy as np
import os
import pandas as pd

from multiprocessing import cpu_count
from skimage import io
from tqdm import tqdm


def data_to_file(data_folder):
    """
    Read dataset folder and create csv of final dataframe.

    Args:
        data_folder (string): Name the folder which contains the data.
    """

    data_folder = os.path.abspath(data_folder)

    sub_folders = os.listdir(data_folder)
    sub_folders.sort()
    grapes_folder = os.path.join(data_folder, sub_folders[0])
    grapes_files = os.listdir(grapes_folder)
    img_folder = os.path.join(data_folder, sub_folders[1])

    file_frame = pd.DataFrame([], columns=['filename', 'target'])
    for fname in os.listdir(img_folder):
        fname_pref = fname[:-4]
        filename = img_folder+'/'+fname

        target = [grapes_folder+'/' +
                  fb for fb in grapes_files if fname_pref in fb]
        file_frame = file_frame.append(
            pd.DataFrame({'filename': filename, 'target': target}))
    file_frame.reset_index(drop=True, inplace=True)

    orig_size = file_frame.shape[0]

    tqdm.pandas()
    res = dd.from_pandas(
        file_frame, npartitions=cpu_count()-1
    ).map_partitions(
        lambda df: df.progress_apply(get_meta, axis=1),
        meta=(None, object)
    ).compute(scheduler='processes')

    res = pd.DataFrame(list(res), columns=['idx_keep', 'colors', 'ratio'])
    idx_keep = res.idx_keep

    nb_disc = idx_keep[idx_keep.eq(False)].shape[0]
    ratio_disc = np.round(nb_disc/orig_size, 2)

    file_frame['color'] = res.colors
    file_frame['visible_ratio'] = res.ratio
    file_frame = file_frame.loc[idx_keep]
    file_frame.reset_index(drop=True, inplace=True)
    save_path = os.path.join(data_folder, 'fname_dataset.json')
    file_frame.to_json(path_or_buf=save_path, orient='split', index=False)

    final_size = file_frame.shape[0]
    ratio = np.round(final_size/orig_size, 2)

    print('Original size of dataframe: {}\n'.format(orig_size),
          'Number of target discarded: {} ({}%)\n'.format(nb_disc, ratio_disc),
          'Final size of dataframe: {} ({}% kept)'.format(final_size, ratio))


def get_meta(row):
    image = io.imread(row['filename'])
    target = io.imread(row['target'])

    tg_col = target.reshape(-1, 3)
    berry_col = np.unique(tg_col, axis=0)
    berry_col.sort(axis=0)
    # pop background color
    berry_col = np.delete(berry_col, 0, 0).reshape(1, 1, 3)

    col_mask = (image == berry_col).astype(int)
    image_mask = (col_mask.sum(-1) == 3).astype(np.uint8)*255
    col_mask = (target == berry_col).astype(int)
    target_mask = (col_mask.sum(-1) == 3).astype(np.uint8)*255

    visible_ratio = np.sum(np.array(image_mask) > 0)
    visible_ratio /= np.sum(np.array(target_mask) > 0)

    if (image_mask.sum() > 0):
        return True, berry_col.flatten(), visible_ratio
    else:
        return False, berry_col.flatten(), visible_ratio


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', '-f', dest='folder',
                        type=str, default='../data/synthetic_grape/',
                        help='Location of synthetic grape dataset folder.')
    args = parser.parse_args()

    data_folder = args.folder
    if 'fname_dataset.json' in os.listdir(data_folder):
        print('JSON file already exists, overwrite it? (y/n)')
        choice = ''
        while choice != 'y' and choice != 'n':
            choice = input()
        if choice == 'y':
            os.remove(os.path.join(data_folder, 'fname_dataset.json'))
            print('File removed, now proceeding to generate new one...')
        elif choice == 'n':
            print('Process aborted.')
            exit(-1)
    data_to_file(data_folder)
