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


class AlabamaDataset(torch.utils.data.Dataset):
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
            divide=2,
            resized_shape=(256, 256)
    ):
        # assert split in self.splits
        self.root = root
        self.divide = divide
        self.resized_shape = resized_shape

        if augment_transform is None:
            self.transform = self.get_default_transform(split, self.resized_shape)
        else:
            self.transform = augment_transform

        self.files = None
        self.divide_width = self.divide_height = 0
        self.load_files(root, split, self.divide)

    @staticmethod
    def get_default_transform(split, resized_shape):
        if split == "train":
            return A.Compose([
                A.ShiftScaleRotate(shift_limit=0, scale_limit=(-0.5, 0.1), rotate_limit=10),
                A.RandomRotate90(),
                A.RandomGamma(),
                A.RGBShift(p=0.2),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                A.Resize(resized_shape[0], resized_shape[1]),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ],
                additional_targets={
                    'mask': 'mask',
                    'border_mask': 'mask'
                }
            )
        else:
            return A.Compose([
                A.Resize(resized_shape[0], resized_shape[1]),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ],
                additional_targets={
                    'mask': 'mask',
                    'border_mask': 'mask'
                }
            )

    def load_files(self, root: str, split: str, divide):
        image_files = open(os.path.join(root, split + ".txt"), "r")
        files = []

        images = sorted([os.path.basename(image.rstrip()) for image in image_files.readlines()])
        image_folder = os.path.join(root, "image")
        mask_folder = os.path.join(root, "mask")
        for image in images:
            image_path = os.path.join(image_folder, image)
            mask_path = os.path.join(mask_folder, image)

            files += [
                dict(image=image_path, mask=mask_path, divide=i)
                for i in range(divide * divide)
            ]

        image = np.array(Image.open(files[0]["image"]))

        self.divide_height = int(image.shape[0]/self.divide)
        self.divide_width = int(image.shape[1]/self.divide)
        self.files = files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict:
        """ Returns a dict containing x, mask
        x: (2, 13, h, w)
        build_mask: (1, h, w)
        demolish_mask: (1, h, w)
        """
        files = self.files[idx]
        x1, x2, y1, y2 = self.get_coord(files['divide'])
        image = np.asarray(Image.open(files["image"]))[x1:x2, y1:y2, :3]
        mask = ((np.array(Image.open(files["mask"])) == 0)*255)[x1:x2, y1:y2]
        mask = create_multiclass_mask(mask, False)

        sample = {
            'image': image,
            'mask': mask[0, ...],
            'border_mask': mask[1, ...]
        }

        transformed = self.transform(**sample)

        image = transformed['image']
        mask = torch.zeros(2, self.resized_shape[0], self.resized_shape[1])
        # print("mask shape" + str(mask.shape))
        # print("transformed['mask'] shape " + str(transformed['mask'].shape))
        # print("transformed['border_mask'] shape " + str(transformed['border_mask'].shape))
        mask[0, ...] = transformed['mask']
        mask[1, ...] = transformed['border_mask']

        return dict(image=image.float(), mask=mask.float())

    def get_resized_coord(self, divide):
        row = int(divide/self.divide)
        col = divide - row*self.divide

        x1 = self.resized_shape[0]*row
        x2 = self.resized_shape[0]*(row + 1)
        y1 = self.resized_shape[1]*col
        y2 = self.resized_shape[1]*(col + 1)

        return x1, x2, y1, y2

    def get_coord(self, divide):
        row = int(divide/self.divide)
        col = divide - row*self.divide

        x1 = self.divide_height*row
        x2 = self.divide_height*(row + 1)
        y1 = self.divide_width*col
        y2 = self.divide_width*(col + 1)

        return x1, x2, y1, y2

    def get_full_resized_shape(self):
        return self.resized_shape[0]*self.divide, self.resized_shape[1]*self.divide