import numpy as np
import os
from PIL import Image
from dataset.utils import *


def save_mask_and_contour(mask, contour, palette, filepath):
    seed = ((mask * (1 - contour)) > 0.5).astype(np.uint8)
    labels = label_watershed(mask, seed)
    im = Image.fromarray(labels.astype(np.uint8), mode='P')
    im.putpalette(palette)
    im.save(filepath)