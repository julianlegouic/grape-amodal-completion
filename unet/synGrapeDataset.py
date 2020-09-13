"""
    Author: Julian LE GOUIC
    Python version: 3.6.7

    IMP Lab - Osaka Prefecture University.
"""

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as tf

from unet.data_aug import MyDataAug
from unet.data_aug import MyRotate
from unet.data_aug import MyHFlip
from unet.data_aug import MyVFlip
from unet.data_aug import ToTensor

from skimage import io
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset

from unet.utils import find_center


class SynGrapeDataset(Dataset):
    """Synthesized grape dataset."""

    def __init__(self, data, transform=None, data_aug=None):
        """
        Args:
            data (pd.DataFrame): DataFrame containing images path.
            transform (torchvision.transforms, optional): Optional
                transforms function to apply on the samples.
            data_aug (object, optional): Optional data augmentation
                to apply on the samples.
        """

        self.data = data

        if transform:
            self.transform = tf.Compose([
                tf.ToPILImage(),
                # 3 = bilinear PIL filter
                tf.Resize((225, 325), 3)
            ])
        else:
            self.transform = tf.ToPILImage()

        if data_aug:
            self.data_aug = MyDataAug(
                proba=0.5,
                transforms=[
                    # 3 = bilinear PIL filter
                    MyRotate(90, 3),
                    MyHFlip(),
                    MyVFlip()
                ]
            )
        else:
            self.data_aug = data_aug

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = io.imread(self.data.iloc[idx, 0])
        target = io.imread(self.data.iloc[idx, 1])
        berry_col = self.data.iloc[idx, 2]
        berry_col = np.array(berry_col).reshape(1, 1, 3)

        # Get the berry mask from input image and target image
        # corresponding to the berry color (berry_col).
        # Scaling by 255 to counter normalization of ToTensor() transform.
        col_mask = (image == berry_col).astype(int)
        image = (col_mask.sum(-1) == 3).astype(np.uint8)*255
        image = image[:, :, np.newaxis]
        col_mask = (target == berry_col).astype(int)
        target = (col_mask.sum(-1) == 3).astype(np.uint8)*255
        target = target[:, :, np.newaxis]

        sample = {
            'image': image, 'target': target,
            'color': berry_col
        }

        sample['image'] = self.transform(sample['image'])
        sample['target'] = self.transform(sample['target'])

        if self.data_aug:
            sample = self.data_aug(sample, shuffle=True, choice=False)

        for key in ['image', 'target']:
            cX, cY = find_center(
                np.uint8(np.array(sample[key]) > 0),
                key
            )

            sample[key+'_center'] = np.array([cX, cY])

        sample = ToTensor()(sample)

        return sample


def data_split(data_file):
    """
    Read dataset file and return training, validation and test set.

    Args:
        data_file (string): Name of the json file which contains the data.
    """

    file_frame = pd.read_json(data_file, orient='split')

    # train/val/test ratios
    train_ratio, test_ratio = 0.85, 0.15
    val_ratio = test_ratio/train_ratio

    train_val_df, test_set = train_test_split(
        file_frame,
        test_size=test_ratio,
        random_state=42
    )

    train_set, val_set = train_test_split(
        train_val_df,
        test_size=val_ratio,
        random_state=42
    )

    return train_set, val_set, test_set
