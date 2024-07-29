from itertools import combinations, product

# do not make tqdm a hard dependency
try:
    import tqdm
except ImportError:
    pass

import numpy as np
from numpy.linalg import qr
from scipy.spatial import kdtree
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

    kd = kdtree.KDTree(points)
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


def descriptor_local_qr(points, n_neighbors=3, redundancy=0, scale_invariant=True, progress_bar=False):

    """
    Generate geometric descriptors from a set of n-dimensional points.
    Uses QR decomposition to find an invariant basis and express relative positions of neighbors in that basis.
    Those values (R) are used as the descriptor of a point.

    If redundancy is > 0, all subsets of size n+1 of the n+redundancy+1 closest neighbours
    will be considered and multiple descriptors per point returned.
    """

    kd = kdtree.KDTree(points)
    descs = []
    idxes = []

    # upper triangular indices
    triag_idxes = list(map( lambda x : x[1] >= x[0], product(range(len(points[0])), range(n_neighbors))))

    worklist = tqdm.tqdm(list(enumerate(list(points)))) if progress_bar else enumerate(list(points))
    for i,p in worklist:
        try:

            # query neighbors, get relative coords by subtracting first result (self)
            _, ix = kd.query(p, n_neighbors+1+redundancy)
            rel_coords = points[ix[1:]] - p
            rel_coords = list(rel_coords)

            for c in combinations(rel_coords, n_neighbors):

                # QR decomposition, we are interested in R (i.e. coordinates of neighbors in local basis Q) 
                a = np.stack(c, axis=1)
                _, R = qr(a)

                # extract upper triangular part
                desc = R.ravel()[triag_idxes]

                # to make scale_invariant we divide by the first entry (i.e. distance to nearest neighbor)
                if scale_invariant:
                    desc /= desc[0]
                    desc = desc[1:]

                descs.append(desc)
                idxes.append(i)

        except RuntimeWarning:
            pass
        except IndexError:
            raise

    return np.array(descs), np.array(idxes)