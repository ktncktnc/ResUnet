import numpy as np
from scipy.ndimage import label, binary_erosion
from skimage import io
from skimage import measure
from skimage.morphology import dilation, square
from skimage.segmentation import watershed
import scipy.ndimage as ndimage

def create_separation(labels):
    tmp = dilation(labels > 0, square(12))
    tmp2 = watershed(tmp, labels, mask=tmp, watershed_line=True) > 0
    tmp = tmp ^ tmp2
    tmp = dilation(tmp, square(3))

    props = measure.regionprops(labels)

    msk1 = np.zeros_like(labels, dtype='bool')

    for y0 in range(labels.shape[0]):
        for x0 in range(labels.shape[1]):
            if not tmp[y0, x0]:
                continue
            if labels[y0, x0] == 0:
                sz = 5
            else:
                sz = 7
                if props[labels[y0, x0] - 1].area < 300:
                    sz = 5
                elif props[labels[y0, x0] - 1].area < 2000:
                    sz = 6
            uniq = np.unique(labels[max(0, y0 - sz):min(labels.shape[0], y0 + sz + 1),
                             max(0, x0 - sz):min(labels.shape[1], x0 + sz + 1)])
            if len(uniq[uniq > 0]) > 1:
                msk1[y0, x0] = True
    return msk1


def create_multiclass_mask(mask, is_contact_points=False):
    num_channels = 2 + (is_contact_points == 1)

    labels, ships_num = label(mask)
    final_mask = np.zeros((num_channels, labels.shape[0], labels.shape[1]))

    if ships_num > 0:
        for i in range(1, ships_num + 1):
            ship_mask = np.zeros_like(labels, dtype='bool')
            ship_mask[labels == i] = 1

            area = np.sum(ship_mask)
            if area < 200:
                contour_size = 1
            elif area < 500:
                contour_size = 2
            else:
                contour_size = 3

            eroded = binary_erosion(ship_mask, iterations=contour_size)
            countour_mask = ship_mask ^ eroded
            final_mask[0, ...] += ship_mask
            final_mask[1, ...] += countour_mask

        if is_contact_points:
            final_mask[2, ...] = create_separation(labels)

    return final_mask

def label_watershed(before, after, component_size=20):
    markers = ndimage.label(after)[0]

    labels = watershed(-before, markers, mask=before, connectivity=8)
    unique, counts = np.unique(labels, return_counts=True)

    for (k, v) in dict(zip(unique, counts)).items():
        if v < component_size:
            labels[labels == k] = 0
    return labels