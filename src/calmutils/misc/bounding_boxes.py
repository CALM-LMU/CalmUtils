from math import ceil, floor

import numpy as np

def get_overlap_bounding_box(length_1, length_2, offset_1=None, offset_2=None):

    # if no offsets are given, assume all zero
    if offset_1 is None:
        offset_1 = [0] * len(length_1)
    if offset_2 is None:
        offset_2 = [0] * len(length_2)

    res_min = []
    res_max = []

    for d in range(len(length_1)):

        min_1 = offset_1[d]
        min_2 = offset_2[d]
        max_1 = min_1 + length_1[d]
        max_2 = min_2 + length_2[d]

        min_ol = max(min_1, min_2)
        max_ol = min(max_1, max_2)

        # no overlap in any one dimension -> return None
        if max_ol < min_ol:
            return None

        res_min.append(min_ol)
        res_max.append(max_ol)

    return res_min, res_max


def get_image_overlaps(img1, img2, off_1=None, off_2=None):
    """
    get overlapping areas of two images with optional offsets

    Parameters
    ----------
    img1, img2: arrays
        input images
    off_1, off_2: lists of int
        offsets of the two images
        optional, we assume no offset if not provided

    Returns
    -------
    r_min_1, r_max_1, r_min_2, r_max_2: lists of int
        minima and maxima of bounding box in local coordinates of each image

    """

    if off_1 is None:
        off_1 = [0] * len(img1.shape)

    if off_2 is None:
        off_2 = [0] * len(img1.shape)

    r_min_1 = []
    r_max_1 = []
    r_min_2 = []
    r_max_2 = []

    for d in range(len(img1.shape)):
        min_1 = off_1[d]
        min_2 = off_2[d]
        max_1 = min_1 + img1.shape[d]
        max_2 = min_2 + img2.shape[d]

        min_ol = max(min_1, min_2)
        max_ol = min(max_1, max_2)

        if max_ol <= min_ol:
            return None

        r_min_1.append(floor(min_ol - off_1[d]))
        r_min_2.append(floor(min_ol - off_2[d]))
        r_max_1.append(ceil(max_ol - off_1[d]))
        r_max_2.append(ceil(max_ol - off_2[d]))

    return r_min_1, r_max_1, r_min_2, r_max_2


def get_iou(bbox1, bbox2):
    (min1, len1) = bbox1
    (min2, len2) = bbox2

    overlap = get_overlap_bounding_box(len1, len2, min1, min2)

    # no overlap
    if overlap is None:
        return 0

    r_min, r_max = overlap

    len_ol = np.array(r_max, dtype=float) - np.array(r_min, dtype=float)
    area_o = np.prod(len_ol)
    area_u = np.prod(len1) + np.prod(len2) - area_o

    return area_o / area_u