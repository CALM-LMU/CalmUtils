from functools import reduce
from itertools import tee, product
from concurrent.futures import ThreadPoolExecutor

from ..phase_correlation import get_axes_aligned_bbox
from scipy.interpolate import RegularGridInterpolator

import numpy as np
from tqdm import tqdm

def subsample_image(image, ds_factor=2):
    image_ds = reduce(lambda arr, i : np.take(arr, np.arange(0, arr.shape[i], ds_factor), axis=i),
                  range(image.ndim), image)
    return image_ds

def pairwise(iterable):
    """
    from itertools recipes
    s -> (s0,s1), (s1,s2), (s2, s3), ...
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def range_includelast(min_, max_, step=1):
    """
    range generator, will always include max_, even if last step is smaller
    (0, 10, 8) -> (0, 8, 10)
    """
    c = min_
    while c < max_:
        yield c
        c += step
    yield max_


def _get_default_bbox_for_fusion(images, transformations):
    mins, maxs = get_axes_aligned_bbox([img.shape for img in images], transformations)
    mins = np.floor(mins).astype(int)
    maxs = np.ceil(maxs).astype(int)
    bbox = list(zip(mins, maxs))
    return bbox


def fuse_image_blockwise(images, transformations, bbox=None, weights=None, oob_val=0, block_size=None, dtype=None, interpolation_mode='nearest', progress_bar=True):

    # default bounding box around all transformed images
    if bbox is None:
        bbox = _get_default_bbox_for_fusion(images, transformations)

    # no blocking necessary
    if block_size is None:
        return fuse_image(images, transformations, bbox, weights, oob_val, dtype, interpolation_mode)

    # allocate final array
    res = np.zeros(tuple((ma_ - mi_ for mi_, ma_ in bbox)), dtype=dtype if dtype is not None else images[0].dtype)

    # list of sub-bboxes
    p = product(*[pairwise(range_includelast(_mi, _ma, s)) for (_mi, _ma), s in zip(bbox, block_size)])
    p = list(p)

    # do multi-threaded
    tpe = ThreadPoolExecutor()
    futures = [tpe.submit(fuse_image, images, transformations, bbox_, weights, oob_val, dtype, interpolation_mode) for bbox_ in p]

    # paste to result, take global offset into account
    iteration_seq = tqdm(zip(futures, p), total=len(futures)) if progress_bar else zip(futures, p)
    for f, bbox_ in iteration_seq:
        res[tuple((slice(mi_-mig_, ma_-mig_) for (mi_, ma_), (mig_, mag_) in zip(bbox_, bbox)))] = f.result()

    tpe.shutdown()
    return res


def _dummy_constant_array(shape, fill_value=0, dtype=None):
    """
    create a dummy constant array of given shape from a single value with 0 stride(s)
    """
    return np.lib.stride_tricks.as_strided(np.full(1, fill_value, dtype=dtype), shape=shape, strides=(0,)*len(shape))


def fuse_image(images, transformations, bbox=None, weights=None, oob_val=0, dtype=None, interpolation_mode='nearest'):

    # default bounding box around all transformed images
    if bbox is None:
        bbox = _get_default_bbox_for_fusion(images, transformations)

    # shape of output
    out_shape = tuple(ma - mi for mi, ma in bbox)

    # ensure we have a list of imgs, transforms even if just one given
    if not isinstance(transformations, list):
        transformations = [transformations]
    if not isinstance(images, list):
        images = [images]

    # unit weight if none given, make sure we have a list of weights
    if weights is None:
        weights = [_dummy_constant_array(img_i.shape, 1.0) for img_i in images]
    if not isinstance(weights, list):
        weights = [weights]

    # prepare output weights
    res_w = np.zeros(out_shape) # TODO: dtype?
    # NB: do this in float, otherwise we might get rounding errors
    res = np.zeros(out_shape) #, dtype=dtype if dtype is not None else images[0].dtype)

    # iter images, weights, transforms
    for (img, weight, mat) in zip(images, weights, transformations):

        # check which rectangular part of final image is affected by "pasting" transformed input
        mins_i, maxs_i = get_axes_aligned_bbox([img.shape], [mat])
        mins_i = np.max([mins_i.astype(int), [mi for mi, _ in bbox]], axis=0)
        maxs_i = np.min([maxs_i.astype(int), [ma for _, ma in bbox]], axis=0)

        # no update necessary
        if any(mi>=ma for mi, ma in zip(mins_i,maxs_i)):
            continue

        # slices into output array
        slices = tuple(slice(mi-gm,ma-gm) for mi,ma,gm in zip(mins_i, maxs_i, [mi for mi, _ in bbox]))

        # transform coords of patch
        coords_i = np.meshgrid(*[np.arange(mi,ma) for mi,ma in zip(mins_i, maxs_i)], indexing='ij')
        # augment coords, apply transform, remove augmented again
        coords_i = np.stack(tuple(coords_i) + (np.ones(coords_i[0].shape, dtype=coords_i[0].dtype),), -1)
        coords_i_tr = coords_i @ np.linalg.inv(mat).transpose()
        coords_i_tr = np.take(coords_i_tr, range(img.ndim), -1)

        # interpolator into image and weight arrays
        img_interp = RegularGridInterpolator(tuple(np.arange(s) for s in img.shape), img, bounds_error=False, fill_value=oob_val, method=interpolation_mode)
        weight_interp = RegularGridInterpolator(tuple(np.arange(s) for s in weight.shape), weight, bounds_error=False, fill_value=0, method=interpolation_mode)

        # get values / weights, add to result arrays
        vals = img_interp(coords_i_tr)
        w = weight_interp(coords_i_tr)
        res[slices] += vals * w
        res_w[slices] += w

    # only divide by nonzero
    res[res_w != 0] = res[res_w != 0] / res_w[res_w != 0]        
    # set zero weight to oob_val
    res[res_w == 0] = oob_val

    return res.astype(dtype if dtype is not None else images[0].dtype)