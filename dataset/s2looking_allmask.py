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
            divide=2,
            resized_shape=(256, 256),
            without_mask=False,
            with_prob=False
    ):
        # assert split in self.splits
        self.width = None
        self.height = None
        self.root = root
        self.divide = divide
        self.resized_shape = resized_shape
        self.without_mask = without_mask
        self.with_prob = with_prob

        if augment_transform is None:
            self.transform = self.get_default_transform(split, self.resized_shape)
        else:
            self.transform = augment_transform
        self.prob_augment = self.get_probs_transform(self.resized_shape)

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
                A.Blur(blur_limit=5, p=0.4),
                A.RGBShift(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.7),
                A.Resize(resized_shape[0], resized_shape[1]),
                PerImageStandazation(),
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
            return A.Compose([
                A.Resize(resized_shape[0], resized_shape[1]),
                PerImageStandazation(),
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

    @staticmethod
    def get_probs_transform(resized_shape):
        return A.Compose([
            A.Resize(resized_shape[0], resized_shape[1]),
            PerImageStandazation(),
            ToTensorV2()
        ],
            additional_targets={
                'image0': 'image'
            })

    def load_files(self, root: str, split: str, divide):
        files = []
        images = glob(os.path.join(root, split, "Image1", "*.png"))
        images = sorted([os.path.basename(image) for image in images])
        for image in images:
            image1 = os.path.join(root, split, "Image1", image)
            image2 = os.path.join(root, split, "Image2", image)
            mask = os.path.join(root, split, "label", image)
            mask1 = os.path.join(root, split, "label1", image)
            mask2 = os.path.join(root, split, "label2", image)
            prob1 = os.path.join(root, split, "prob_img1", "prob_" + image[:-4] + ".npz")
            prob2 = os.path.join(root, split, "prob_img2", "prob_" + image[:-4] + ".npz")

            files += [
                dict(image1=image1, image2=image2, mask=mask, mask1=mask1, mask2=mask2, divide=i, prob1=prob1, prob2=prob2)
                for i in range(divide * divide)
            ]

        image1 = np.array(Image.open(files[0]["image1"]))

        self.divide_height = int(image1.shape[0]/self.divide)
        self.divide_width = int(image1.shape[1]/self.divide)
        self.height, self.width = image1.shape[0], image1.shape[1]
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

        image1 = np.asarray(Image.open(files["image1"]))[x1:x2, y1:y2, ...]
        image2 = np.array(Image.open(files["image2"]))[x1:x2, y1:y2, ...]

        if not self.without_mask:
            mask = (np.array(Image.open(files["mask"])) / 255.0)[x1:x2, y1:y2]
            mask1 = np.asarray(Image.open(files["mask1"]))[x1:x2, y1:y2, ...]
            mask2 = np.asarray(Image.open(files["mask2"]))[x1:x2, y1:y2, ...]
            if len(mask1.shape) == 3 and mask1.shape[-1] > 1:
                mask1 = mask1[:, :, 2]
            if len(mask2.shape) == 3 and mask2.shape[-1] > 1:
                mask2 = mask2[:, :, 0]
            mask1 = create_multiclass_mask(mask1, False)
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
        else:
            sample = {
                'image': image1,
                'image0': image2
            }

        transformed = self.transform(**sample)
        image1 = transformed['image']
        image2 = transformed['image0']

        if self.with_prob:
            prob1 = np.load(files['prob1'])['a'][x1:x2, y1:y2, np.newaxis]
            prob2 = np.load(files['prob2'])['a'][x1:x2, y1:y2, np.newaxis]
            prob_sample = {
                'image': prob1,
                'image0': prob2
            }
            prob_transformed = self.prob_augment(**prob_sample)
            image1 = torch.cat((image1, prob_transformed['image']), dim=0)
            image2 = torch.cat((image2, prob_transformed['image0']), dim=0)

        if not self.without_mask:
            mask = transformed['mask']

            mask1 = torch.zeros(2, mask.shape[0], mask.shape[1])
            mask1[0, ...] = transformed['mask1']
            mask1[1, ...] = transformed['border_mask1']

            mask2 = torch.zeros(2, mask.shape[0], mask.shape[1])
            mask2[0, ...] = transformed['mask2']
            mask2[1, ...] = transformed['border_mask2']

            return dict(x=image1.float(), y=image2.float(), mask=mask.float(), mask1=mask1.float(), mask2=mask2.float(), path=files["mask"])
        else:
            return dict(x=image1.float(), y=image2.float(), path=files["mask"])

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
        if row < self.divide - 1:
            x2 = self.divide_height*(row + 1)
        else:
            x2 = self.height

        y1 = self.divide_width*col
        if col < self.divide - 1:
            y2 = self.divide_width*(col + 1)
        else:
            y2 = self.width

        return x1, x2, y1, y2

    def get_full_resized_shape(self):
        return self.resized_shape[0]*self.divide, self.resized_shape[1]*self.divide

