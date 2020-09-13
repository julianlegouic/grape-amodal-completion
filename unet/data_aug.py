"""
    Author: Julian LE GOUIC
    Python version: 3.6.7

    IMP Lab - Osaka Prefecture University.
"""

import numpy as np
import random
import torchvision.transforms.functional as tff


class MyDataAug(object):
    """Personalized class for data augmentation."""

    def __init__(self, proba, transforms):
        """
        Args:
            proba (float): probability threshold to apply transformations
                in the pipeline.
            transforms (list): list of transformations to apply for data
                augmentation.
        """
        self.proba = proba

        self.transforms = transforms

    def __call__(self, sample, shuffle=True, choice=False):
        if shuffle:
            random.shuffle(self.transforms)
        if choice:
            t = random.choice(self.transforms)
            return t(sample)
        else:
            if random.random() > self.proba:
                for t in self.transforms:
                    sample = t(sample)
        return sample


class MyRotate(object):
    """Personalized class for rotation transformation."""

    def __init__(self, degrees, resample):
        degrees = (-degrees, degrees)
        self.angle = random.uniform(degrees[0], degrees[1])
        self.resample = resample

    def __call__(self, sample):
        sample['image'] = tff.rotate(
            sample['image'], self.angle, self.resample, fill=(0,))
        sample['target'] = tff.rotate(
            sample['target'], self.angle, self.resample, fill=(0,))
        return sample


class MyHFlip(object):
    """Personalized class for horizontal flip transformation."""

    def __call__(self, sample):
        sample['image'] = tff.hflip(sample['image'])
        sample['target'] = tff.hflip(sample['target'])
        return sample


class MyVFlip(object):
    """Personalized class for vertical flip transformation."""

    def __call__(self, sample):
        sample['image'] = tff.vflip(sample['image'])
        sample['target'] = tff.vflip(sample['target'])
        return sample


class ToTensor(object):
    """Convert sample object values in Tensors."""

    def __call__(self, sample, t=0.5):
        # Due to transformations, some pixels value are inbetween 0 and 255.
        # t is the threshold from where we consider pixels as grape or bg
        image = (np.array(sample['image']) > t*255).astype(np.uint8)*255
        target = (np.array(sample['target']) > t*255).astype(np.uint8)*255
        sample['image'] = tff.to_tensor(image)
        sample['target'] = tff.to_tensor(target)

        return sample
