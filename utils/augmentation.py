import math
import time
import warnings
import cv2
from typing import Dict, Any

warnings.simplefilter("ignore", UserWarning)
from timeit import default_timer as timer

from skimage import transform
import random
from albumentations.core.transforms_interface import DualTransform, ImageOnlyTransform
import numpy as np


class PerImageStandazation(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super().__init__(always_apply, p)

    def apply(self, image: np.ndarray, **params):
        image = image.astype(np.float32)
        num_pixels = np.prod(image.shape[-3:])
        image_mean = np.mean(image, axis=(-1, -2, -3), dtype=np.float32 )

        stddev = np.std(image, axis=(-1, -2, -3))
        min_stddev = 1.0/np.sqrt(num_pixels)
        adjusted_stddev = np.maximum(stddev, min_stddev)

        image -= image_mean
        image = np.divide(image, adjusted_stddev)
        return image


class RandomCropSaveSegmentMask(DualTransform):
    """Crop a random part of the input.
    Args:
        height (int): height of the crop.
        width (int): width of the crop.
        p (float): probability of applying the transform. Default: 1.
    Targets:
        image, mask, bboxes, keypoints
    Image types:
        uint8, float32
    """

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def __init__(self, height, width, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.height = height
        self.width = width
        self.patches = None
        self.patch = None

    def apply(self, img, h_start=0, w_start=0, **params):
        return img[self.patch['x1']:self.patch['x2'], self.patch['y1']:self.patch['y2']]

    def get_params(self):
        return {"h_start": random.random(), "w_start": random.random()}

    def get_best_patch(self, mask):
        sum_over_height = np.sum(mask, 0, dtype=float)
        sum_over_width = np.sum(mask, 1, dtype=float)

        conv_sum_over_height = np.convolve(sum_over_height, np.ones(self.width, dtype=float), mode='valid')
        conv_sum_over_width = np.convolve(sum_over_width, np.ones(self.height, dtype=float), mode='valid' )
        total_sum_height = np.sum(conv_sum_over_height)
        total_sum_width = np.sum(conv_sum_over_width)

        if total_sum_height == 0 or total_sum_width == 0:
            rand_width = np.random.choice(conv_sum_over_height.size)
            rand_height = np.random.choice(conv_sum_over_width.size)
        else:
            rand_width = np.random.choice(conv_sum_over_height.size, p=conv_sum_over_height/total_sum_height)
            rand_height = np.random.choice(conv_sum_over_width.size, p=conv_sum_over_width/total_sum_width)

        self.patch = {
            "x1": rand_height,
            "x2": rand_height + self.height,
            "y1": rand_width,
            "y2": rand_width + self.width
        }


class RescaleTarget(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, tuple):
            self.output_size = int(np.random.uniform(output_size[0], output_size[1]))
        else:
            self.output_size = output_size

    def __call__(self, sample):
        sat_img, map_img = sample["sat_img"], sample["map_img"]

        h, w = sat_img.shape[:2]

        if h > w:
            new_h, new_w = self.output_size * h / w, self.output_size
        else:
            new_h, new_w = self.output_size, self.output_size * w / h

        new_h, new_w = int(new_h), int(new_w)

        # change the range to 0-1 rather than 0-255, makes it easier to use sigmoid later
        sat_img = transform.resize(sat_img, (new_h, new_w))

        map_img = transform.resize(map_img, (new_h, new_w))

        return {"sat_img": sat_img, "map_img": map_img}


class RandomRotationTarget(object):
    """Rotate the image and target randomly in a sample.

    Args:
        degrees (tuple or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resize (boolean): Expand the image to fit
    """

    def __init__(self, degrees, resize=False):
        if isinstance(degrees, int):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if isinstance(degrees, tuple):
                raise ValueError("Degrees needs to be either an int or tuple")
            self.degrees = degrees

        assert isinstance(resize, bool)

        self.resize = resize
        self.angle = np.random.uniform(self.degrees[0], self.degrees[1])

    def __call__(self, sample):

        sat_img = transform.rotate(sample["sat_img"], self.angle, self.resize)
        map_img = transform.rotate(sample["map_img"], self.angle, self.resize)

        return {"sat_img": sat_img, "map_img": map_img}


class RandomCropTarget(object):
    """
    Crop the image and target randomly in a sample.

    Args:
    output_size (tuple or int): Desired output size. If int, square crop
        is made.

    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):

        sat_img, map_img = sample["sat_img"], sample["map_img"]

        h, w = sat_img.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        sat_img = sat_img[top: top + new_h, left: left + new_w]
        map_img = map_img[top: top + new_h, left: left + new_w]

        return {"sat_img": sat_img, "map_img": map_img}
