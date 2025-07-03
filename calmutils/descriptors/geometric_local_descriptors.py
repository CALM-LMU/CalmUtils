from itertools import combinations

# do not make tqdm a hard dependency
try:
    import tqdm
except ImportError:
    pass

import numpy as np
from numpy.linalg import qr
from scipy.spatial import KDTree
from skimage.transform import AffineTransform


def descriptor_local_2d(points, n_neighbors=3, redundancy=0, scale_invariant=True, progress_bar=False):

    """
    Generate geometric descriptors from a set of points.

    The descriptor for each point are the coordinates of the n+1 closest neighbours,
    rotated and scaled so that the vector between a point and its FIRST closest neighbor
    points along the first axis and has unit length.

    If redundancy is > 0, all subsets of size n+1 of the n+redundancy+1 closest neighbours
    will be considered and multiple descriptors per point returned.

    NOTE: only works for 2D at the moment
    """

    kd = KDTree(points)
    descs = []
    idxes = []

    worklist = tqdm.tqdm(list(enumerate(list(points)))) if progress_bar else enumerate(list(points))
    for i,p in worklist:
        try:
            _, ix = kd.query(p, n_neighbors+2+redundancy)

            rel_coords = points[ix[1:]] - p
            rel_coords = list(rel_coords)

            for c in combinations(rel_coords, n_neighbors+1):

                first = c[0]
                others = c[1:]

                a1 = np.arctan2(*list(reversed(list(first))))

                desc = []

                desc.append(AffineTransform(rotation=-a1)(others)/ np.linalg.norm(first) if scale_invariant else 1)
                desc = np.array(desc).ravel()
                descs.append(desc)
                idxes.append(i)
        except RuntimeWarning:
            pass
    return np.array(descs), idxes


def descriptor_local_qr(points, n_neighbors=3, redundancy=0, scale_invariant=True):

    """
    Generate geometric descriptors from a set of n-dimensional points.
    Uses QR decomposition to find an invariant basis and express relative positions of neighbors in that basis.
    Those values (R) are used as the descriptor of a point.

    If redundancy is > 0, all subsets of size n+1 of the n+redundancy+1 closest neighbours
    will be considered and multiple descriptors per point returned.
    """

    N, d = points.shape

    # mask of upper triangular elements -> to select from QR factorization results
    triu_mask = np.triu(np.ones((d, n_neighbors))).astype(bool)

    # query nearest neighbors with kd-tree
    kd = KDTree(points)
    _, idxes = kd.query(points, n_neighbors + redundancy +1)
    # discard neares neighbor (self)
    idxes = idxes[:, 1:]

    # vectors to nearest neighbors
    rel_coords = points[idxes.T.ravel()].reshape(n_neighbors + redundancy, N, d) - points

    # collect descriptors & indices for redundant combinations of neighbors
    result_descriptors = []
    result_indices = []

    for rel_coords_i in combinations(rel_coords, n_neighbors):

        # transpose to N, d, n_neighbors
        qr_input = np.transpose(rel_coords_i, (1, 2, 0))

        # QR automatically loops over first dimension, with mode="r" we ignore Q
        r = qr(qr_input, mode="r")

        # descriptor: upper triangular elements of R
        desc = r[:, triu_mask]

        if scale_invariant:
            # divide by length of first vector (first element of descriptor)
            desc = (desc.T / desc.T[0]).T
            # discard as it is always 1
            desc = desc[:, 1:]

        result_descriptors.append(desc)
        result_indices.append(np.arange(N, dtype=int))

    return np.vstack(result_descriptors), np.concat(result_indices)