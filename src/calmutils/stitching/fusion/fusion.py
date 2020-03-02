from functools import reduce
from itertools import tee, product
from concurrent.futures import ThreadPoolExecutor

import numpy as np

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
    

def fuse_image(bbox, images, transformations, weights=None, oob_val=0, block_size=None, dtype=None):

    # no blocking necessary
    if block_size is None:
        return fuse_image_(bbox, images, transformations, weights, oob_val, dtype)

    # allocate final array
    res = np.zeros(tuple((ma_ - mi_ for mi_, ma_ in bbox)), dtype=dtype)

    # list of sub-bboxes
    p = product(*[pairwise(range_includelast(_mi, _ma, s)) for (_mi, _ma), s in zip(bbox, block_size)])
    p = list(p)

    # do multi-threaded
    tpe = ThreadPoolExecutor()
    futures = [tpe.submit(fuse_image_, bbox_, images, transformations, weights, oob_val, dtype) for bbox_ in p]

    # paste to result, take global offset into account
    for f, bbox_ in zip(futures, p):
        res[tuple((slice(mi_-mig_, ma_-mig_) for (mi_, ma_), (mig_, mag_) in zip(bbox_, bbox)))] = f.result()

    tpe.shutdown()
    return res

def fuse_image_(bbox, images, transformations, weights=None, oob_val=0, dtype=None):

    # create output coordinates
    coords = np.meshgrid(*[range(mi,ma) for mi,ma in bbox], indexing='ij')
    coords = np.stack(coords + [np.ones(coords[0].shape, dtype=coords[0].dtype)], -1)

    # shape of output + ndim
    init_shape = coords.shape
    coords = coords.reshape((np.prod(coords.shape[:-1]), coords.shape[-1]))

    # ensure we have a list of imgs, transforms even if just one given
    if not isinstance(transformations, list):
        transformations = [transformations]
    if not isinstance(images, list):
        images = [images]

    # unit weight if none given, make sure we have a list of weights
    if weights is None:
        weights = [np.ones(img_i.shape) for img_i in images]
    if not isinstance(weights, list):
        weights = [weights]

    # prepare output, TODO: can we do it more space-efficient?
    res_w = np.zeros((len(images),) + init_shape[:-1]) # TODO: dtype?
    # TODO: do this in float? otherwise we might get rounding errors
    res = np.zeros((len(images),) + init_shape[:-1], dtype=dtype if dtype is not None else images[0].dtype)

    # iter images, transforms
    for (i, mat) in enumerate(transformations):
        coords_i = coords @ np.linalg.inv(mat).transpose()
        coords_i = np.take(coords_i, range(len(bbox)), -1)

        # TODO: interpolation?
        oob_i = np.any((coords_i >= images[i].shape) + (coords_i < (0)*images[i].ndim), -1)
        coords_i = coords_i.astype(np.int)

        idx_i = np.ravel_multi_index(tuple([np.take(coords_i, j, -1) for j in range(images[i].ndim)]),
                               images[i].shape, mode='clip')

        # take flat idxes from original image
        res_i = images[i].flat[idx_i]
        if weights is not None:
            weights_i = weights[i].flat[idx_i]
            weights_i[oob_i] = oob_val
            res_i = res_i * weights_i
            res_w[i] = weights_i.reshape(init_shape[:-1])
        # set oob to predefined value
        res_i[oob_i] = oob_val
        res[i] = res_i.reshape(init_shape[:-1])

    res = np.sum(res, axis=0)
    res_w = np.sum(res_w, axis=0)
    # only divide by nonzero
    res[res_w != 0] = res[res_w != 0] / res_w[res_w != 0]        
    # set zero weight to oob_val
    res[res_w == 0] = oob_val
    return res.astype(dtype if dtype is not None else images[0].dtype)