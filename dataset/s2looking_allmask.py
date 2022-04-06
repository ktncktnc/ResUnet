from glob import glob
from typing import Dict

import albumentations as A
from utils.augmentation import *
import numpy as np
import os
import torch
from PIL import Image
from albumentations.pytorch import ToTensorV2
from dataset.utils import *


class S2LookingAllMask(torch.utils.data.Dataset):
    """ The Satellite Side-Looking (S2Looking) dataset from 'S2Looking: A Satellite Side-Looking
    Dataset for Building Change Detection', Shen at al. (2021)
    https://arxiv.org/abs/2107.09244
    'S2Looking is a building change detection dataset that contains large-scale side-looking
    satellite images captured at varying off-nadir angles. The S2Looking dataset consists of
    5,000 registered bitemporal image pairs (size of 1024*1024, 0.5 ~ 0.8 m/pixel) of rural
    areas throughout the world and more than 65,920 annotated change instances. We provide
    two label maps to separately indicate the newly built and demolished building regions
    for each sample in the dataset.'
    """
    splits = ["train", "val", "test"]

    def __init__(
            self,
            root: str = ".data/s2looking",
            split: str = "train",
            augment_transform=None,
            divide=2
    ):
        # assert split in self.splits
        self.root = root
        self.random_crop = None
        self.divide = divide
        if augment_transform is None:
            targets = {
                'image0': 'image',
                'mask1': 'mask',
                'mask2': 'mask',
                'border_mask1': 'mask',
                'border_mask2': 'mask'
            }
            if split == "train":
                self.random_crop = RandomCropSaveSegmentMask(512, 512, mask_threshold=0.35)
                augment_transform = A.Compose([
                    A.ShiftScaleRotate(shift_limit=0, scale_limit=(-0.5, 0.1), rotate_limit=10),
                    self.random_crop,
                    A.RandomRotate90(),
                    A.RandomGamma(),
                    A.RGBShift(p=0.2),
                    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                    A.Resize(256, 256),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2()
                ],
                    additional_targets={
                        'image0': 'image',
                        'mask1': 'mask',
                        'mask2': 'mask',
                        'border_mask1': 'mask',
                        'border_mask2': 'mask'
                    }
                )
            else:
                self.random_crop = RandomCropSaveSegmentMask(512, 512, mask_threshold=1.0)
                augment_transform = A.Compose([
                    self.random_crop,
                    A.Resize(256, 256),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2()
                ],
                    additional_targets={
                        'image0': 'image',
                        'mask1': 'mask',
                        'mask2': 'mask',
                        'border_mask1': 'mask',
                        'border_mask2': 'mask'
                    }
                )
        self.transform = augment_transform
        self.files, self.image_names = self.load_files(root, split)

    @staticmethod
    def load_files(root: str, split: str, divide):
        files = []
        images = glob(os.path.join(root, split, "Image1", "*.png"))
        images = sorted([os.path.basename(image) for image in images])
        for image in images:
            image1 = os.path.join(root, split, "Image1", image)
            image2 = os.path.join(root, split, "Image2", image)
            mask = os.path.join(root, split, "label", image)
            mask1 = os.path.join(root, split, "label1", image)
            mask2 = os.path.join(root, split, "label2", image)

            files += [
                dict(image1=image1, image2=image2, mask=mask, mask1=mask1, mask2=mask2, divide=i)
                for i in range(divide * divide)
            ]

        return files, images

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict:
        """ Returns a dict containing x, mask
        x: (2, 13, h, w)
        build_mask: (1, h, w)
        demolish_mask: (1, h, w)
        """
        files = self.files[idx]
        image1 = np.array(Image.open(files["image1"]))
        image2 = np.array(Image.open(files["image2"]))

        mask = np.array(Image.open(files["mask"])) / 255.0
        mask = np.expand_dims(mask, axis=2)

        mask1 = np.array(Image.open(files["mask1"]))[..., 2]
        mask1 = create_multiclass_mask(mask1, False)

        mask2 = np.array(Image.open(files["mask2"]))[..., 0]
        mask2 = create_multiclass_mask(mask2, False)

        sample = {
            'image': image1,
            'image0': image2,
            'mask': mask,
            'mask1': mask1[0, ...],
            'mask2': mask2[0, ...],
            'border_mask1': mask1[1, ...],
            'border_mask2': mask2[1, ...]
        }

        self.random_crop.get_best_patch(mask)
        transformed = self.transform(**sample)

        image1 = transformed['image']
        image2 = transformed['image0']
        mask = transformed['mask'][..., 0]

        mask1 = torch.zeros(2, mask.shape[0], mask.shape[1])
        mask1[0, ...] = transformed['mask1']
        mask1[1, ...] = transformed['border_mask1']

        mask2 = torch.zeros(2, mask.shape[0], mask.shape[1])
        mask2[0, ...] = transformed['mask2']
        mask2[1, ...] = transformed['border_mask2']

        return dict(x=image1.float(), y=image2.float(), mask=mask.float(), mask1=mask1.float(), mask2=mask2.float())
