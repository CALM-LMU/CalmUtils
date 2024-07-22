import numpy as np
from calmutils.stitching.phase_correlation import phasecorr_align


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
        minima and maxima of bounding box in local coordiantes of each image

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

        r_min_1.append(min_ol - off_1[d])
        r_min_2.append(min_ol - off_2[d])
        r_max_1.append(max_ol - off_1[d])
        r_max_2.append(max_ol - off_2[d])

    return r_min_1, r_max_1, r_min_2, r_max_2


def get_shift(img1, img2, off_1=None, off_2=None):
    """
    Get the translation between two images via phase correlation in the overlapping area via phase correlation

    Parameters
    ----------
    img1, img2: arrays
        input images
    off_1, off_2: 1d-arrays
        estimated (e.g. metadata offset of images)
        optional, we assume no offset if not provided

    Returns
    -------
    r_off1, r_off2: 1d-arrays
        registered offsets of the images (r_off1 is equal to off_1)
    corr: float
        cross-correlation of the images after registration, may be None if we could not register
    """

    if off_1 is None:
        off_1 = [0] * len(img1.shape)

    if off_2 is None:
        off_2 = [0] * len(img1.shape)

    ol = get_image_overlaps(img1, img2, off_1, off_2)

    # images do not overlap according to their offsets -> return input
    if ol is None:
        return (off_1, off_2, None)

    # ol are two min-max bounding boxes, use to select overlaping regions from images
    r_min_1, r_max_1, r_min_2, r_max_2 = ol
    cut1 = img1[tuple(map(lambda x: slice(*x), zip(r_min_1, r_max_1)))]
    cut2 = img2[tuple(map(lambda x: slice(*x), zip(r_min_2, r_max_2)))]

    shift_cut, corr = phasecorr_align(cut2, cut1)
    ol_registered = get_image_overlaps(img1, img2, off_1, np.array(off_2) - shift_cut)

    # no overlap after registration -> return metadata
    if ol_registered is None:
        return (off_1, off_2, None)

    return np.array(off_1), np.array(off_2) - shift_cut, corr


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
    shape = tuple((maxs - mins).astype(np.int)) + (len(imgs),)

    return shape, mins.astype(np.int)


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


def stitch(ref_img, imgs, ref_off=None, offs=None, cval=-1, corr_thresh=0.5):
    """
    Simple translational registration of a set of images to a reference image
    Only comparisions to reference are done, not within the other images

    Parameters
    ----------
    ref_img: array
        reference image
    imgs: list of arrays
        images to register
    ref_off: 1d-array-like of int
        offset of reference image, optional
    offs: list of 1d-array-like of int
        offsets of images to register, optional
    cval: float
        constant background value, -1 by default
    corr_thresh: float
        minimal corralation coefficient between images after registration,
        if it is below threshold, we will keep metadata

    Returns
    -------
    fused: array
        fused image
    shifts: list of 1d-array-like of int
        registered offsets of all images (reference, img0, img1, ...)
    off: 1d-array-like
        offset of fused image (coordinates of origin)
    corrs: list of float
        correlation coefficients of imgs to ref_img after registration
        will be None if registration failed or was rejected because of threshold

    """

    if ref_off is None:
        ref_off = [0] * len(ref_img.shape)
    shifts = []
    corrs = []
    for i, img in enumerate(imgs):
        _, shift, corr = get_shift(ref_img, img, ref_off, None if offs is None else offs[i])
        if corr is None or corr < corr_thresh:
            shifts.append([0]*len(img.shape) if offs is None else offs[i])
            corrs.append(None)
        else:
            shifts.append(shift)
            corrs.append(corr)

    fus, off = fuse([ref_img] + imgs, [ref_off] + shifts, cval=cval)
    return fus, [ref_off] + shifts, off, corrs