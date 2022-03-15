import numpy as np
import os
from PIL import Image
from dataset.utils import *
import random

def convert_to_color_map(masks, w = 256, h = 256):
    new_masks = torch.zeros(w,h)
    if masks.shape[0] < 1:
        return new_masks.numpy()
    list_color = random.sample(range(30,230), masks.shape[0])
    for i in range(masks.shape[0]):
        tmp = masks[i] * list_color[i]
        new_masks = new_masks + tmp
    return new_masks.numpy()

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
