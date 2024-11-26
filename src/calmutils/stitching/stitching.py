from math import floor, ceil
from itertools import combinations

import numpy as np

from calmutils.stitching.phase_correlation import phasecorr_align
from calmutils.stitching.transform_helpers import translation_matrix
from calmutils.stitching.registration import register_translations


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


def phasecorr_align_overlapping_region(img1, img2, off_1=None, off_2=None, subpixel=False, return_relative_shift=False, min_overlap=None):
    """
    Get the translation between two images via phase correlation in the overlapping area.
    If offsets are provided, we will only process overlapping region for speedup.

    Parameters
    ----------
    img1, img2: arrays
        input images (img1: target, img2: moving)
    off_1, off_2: 1d-arrays
        estimated (e.g. metadata offset of images)
        optional, we assume no offset if not provided
    subpixel: bool
        whether to do subpixel-accurate phase correlation
    return_relative_shift: bool
        whether to return relative shift (to be applied on top of existing offset)
        or absolute shift (including existing offset)

    Returns
    -------
    registered_offset: 1d-arrays
        registered offset of the moving image (img2)
    corr: float
        cross-correlation of the images after registration, may be None if we could not register
    """

    no_offsets_provided = off_1 is None and off_2 is None

    # default to no offsets
    if off_1 is None:
        off_1 = [0] * len(img1.shape)
    if off_2 is None:
        off_2 = [0] * len(img1.shape)

    # check overlap
    ol = get_image_overlaps(img1, img2, off_1, off_2)
    # images do not overlap according to their offsets -> return input
    if ol is None:
        return off_2, None

    # ol are two min-max bounding boxes, use to select overlapping regions from images
    r_min_1, r_max_1, r_min_2, r_max_2 = ol
    cut1 = img1[tuple(map(lambda x: slice(*x), zip(r_min_1, r_max_1)))]
    cut2 = img2[tuple(map(lambda x: slice(*x), zip(r_min_2, r_max_2)))]

    # defaults for minimal overlap
    # when no offsets (overlapping region) are given: min 2.5% of image in all dimensions
    # when we have overlapping regions: min 25% of image
    if min_overlap is None:
        min_overlap = np.array(cut1.shape) * 0.025 if no_offsets_provided else np.array(cut1.shape) * 0.25

    # calculate phasecorr in overlap, add to original offset for absolute shift
    shift_relative, corr = phasecorr_align(cut1, cut2, subpixel=subpixel, min_overlap=min_overlap)
    shift_absolute = np.array(off_2) + shift_relative

    # sanity-check resulting overlap again
    ol_registered = get_image_overlaps(img1, img2, off_1, shift_absolute)
    # no overlap after registration -> return metadata
    if ol_registered is None:
        return off_2, None

    return shift_relative if return_relative_shift else shift_absolute, corr


def get_fused_shape(imgs, offs):
    """
    get the required size of stacked image (fusion is done by projecting along last axis) and offset of origin

    Parameters
    ----------
    imgs: list of arrays
        the images to fuse
    offs: list of 1d-arrays
        offsets of the images

    Returns
    -------
    shape: int-tuple
        required size of fused image (with additional last dimension of size len(imgs))
    off: 1d-array
        offset of the minimum of fused image (coordinates of origin)
    """

    mins = np.array([np.array(off) for off in offs])
    maxs = np.array([np.array(off) + np.array(img.shape) for (off, img) in zip(offs, imgs)])
    mins = np.apply_along_axis(np.min, 0, mins)
    maxs = np.apply_along_axis(np.max, 0, maxs)
    shape = tuple((maxs - mins).astype(int)) + (len(imgs),)

    return shape, mins.astype(int)


def fuse(imgs, offs, fun=np.max, cval=-1):
    """
    fuse images by stacking translated versions and projecting along stack axis

    Parameters
    ----------
    imgs: list of arrays
        the images to fuse
    offs: list of 1d-arrays
        offsets of the images
    fun: callable with `axis`-parameters
        function to project along stack axis
    cval: float
        constant value of "empty" background

    Returns
    -------
    fused: array
        fused image
    mi: 1d-array
        coordinates of origin of fused image

    """
    sh, mi = get_fused_shape(imgs, offs)
    out = np.zeros(sh)
    out += cval
    for idx, ioff in enumerate(zip(imgs, offs)):
        img, off = ioff
        slice_idx = tuple(
            map(lambda x: slice(int(x[1] - x[2]), int(x[0] + x[1] - x[2])), zip(img.shape, list(off), list(mi)))) + (
                    idx,)
        out[slice_idx] = img
    res = fun(out, axis=-1)
    return res, mi


def stitch(images, positions=None, corr_thresh=0.5, subpixel=False, return_shift_vectors=False, reference_idx=0):

    # when no positions are given, assume all images at origin (will check all possible pairs)
    if positions is None:
        positions = [np.zeros(images[0].ndim)] * len(images)

    # pairwise transform estimation
    global_registration_input = {}
    for idx1, idx2 in combinations(range(len(images)), 2):

        img1, img2 = images[idx1], images[idx2]
        position1, position2 = positions[idx1], positions[idx2]
        shift, corr = phasecorr_align_overlapping_region(img1, img2, position1, position2, subpixel=subpixel,
                                                         return_relative_shift=True)

        # we have a shift with good enough correlation
        if corr is not None and corr > corr_thresh:
            # make dummy "point matches" from image corners for global registration
            # it does not really matter which points we use as long as they don't lie along the same axis
            corners = np.stack(np.meshgrid(*((0, s) for s in img1.shape), indexing='ij'), -1).reshape(
                -1, img1.ndim).astype(float)

            # shift "matched points" in img2
            corners_shifted = corners - shift

            # add image pair and matches to global registration input
            global_registration_input[(idx1, idx2)] = (corners, corners_shifted)

    # run global registration, result is idx -> transformation matrix
    if len(global_registration_input) > 0:
        # use either reference idx or the first index present in pairwise results as fixed
        if any(idx1 == reference_idx or idx2 == reference_idx for (idx1, idx2) in global_registration_input.keys()):
            fixed_idx_global_opt = reference_idx
        else:
            fixed_idx_global_opt = next((idx1 for (idx1, idx2) in global_registration_input.keys()))
        global_registration_results = register_translations(global_registration_input, [fixed_idx_global_opt])
    else:
        global_registration_results = {}

    # map to list, re-add missing
    transforms = []
    for idx in range(len(images)):
        # idx is missing from registration results -> use original offset (e.g. metadata)
        if not idx in global_registration_results:
            transforms.append(translation_matrix(-positions[reference_idx] + positions[idx]))
        else:
            # append local shift result to global offset
            tr = global_registration_results[idx]
            tr = translation_matrix(-positions[reference_idx] + positions[idx]) @ tr
            transforms.append(tr)

    # return either just the shift vectors or full transformation matrices
    if return_shift_vectors:
        return [tr[:-1, -1] for tr in transforms]
    else:
        return transforms