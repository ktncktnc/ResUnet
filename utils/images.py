import numpy as np
import os
from PIL import Image
from dataset.utils import *


def save_mask_and_contour(mask, contour, palette, filepath):
    seed = ((mask * (1 - contour)) > 0.5).astype(np.uint8)
    labels = label_watershed(mask, seed).astype(np.uint8)

    im = Image.fromarray(labels, mode='P')
    im.putpalette(palette)
    im.save(filepath)

    return multiclass_mask_to_multichannel(labels)


def multiclass_mask_to_multichannel(mask):
    nunique = np.unique(mask, return_counts=False)

    result_mask = np.array([mask == v for v in nunique[1:]])
    return result_mask