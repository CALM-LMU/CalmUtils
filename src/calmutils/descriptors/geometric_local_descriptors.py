from itertools import combinations

# do not make tqdm a hard dependency
try:
    import tqdm
except ImportError:
    pass

import numpy as np
from scipy.spatial import kdtree
from skimage.transform import AffineTransform

def descriptor_local_2d(points, n_neighbors=3, redundancy=0, scale_invariant=True, progress_bar=True):

    """
    Generate geometric descriptors from a set of points.
    
    The descriptor for each point are the coordinates of the n+1 closest neighbours,
    rotated and scaled so that the vector between a point and its FIRST closest neighbor
    points along the first axis and has unit length.
    
    If redundancy is > 0, all subsets of size n+1 of the n+redundancy+1 closest neighbours
    will be considered and multiple descriptors per point returned.
    
    NB: only works for 2D at the moment
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
        except RuntimeWarning as e:
            pass
    return np.array(descs), idxes