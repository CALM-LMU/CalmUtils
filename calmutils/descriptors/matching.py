import numpy as np
from scipy.spatial import KDTree

def match_descriptors_kd(descriptors1, descriptors2, max_ratio=1.0, cross_check=True):
    """
    Match two sets of descriptors using KD-Trees (only works with Euclidean distance)
    Follows conventions of skimage.feature.match_descriptors
    """

    # query 2 closest neighbors in descriptors2 for every data point in descriptors1 using kd-tree
    kd = KDTree(descriptors2)
    ds, idxs = kd.query(descriptors1, k=2)

    # stack all indices in descriptors1 set with closest index in descriptors2
    matches = np.stack([np.arange(len(descriptors1), dtype=int), idxs.T[0]], -1)

    if cross_check:
        # find indices of closest match of descriptors2 to descriptors1
        kd_cross = KDTree(descriptors1)
        _, idxs_cross = kd_cross.query(descriptors2)

        # check the cross indices for each match -> they should be 0, 1, 2, ...
        selection = idxs_cross[matches.T[1]] == np.arange(len(matches), dtype=int)

        # discard matches and distances that do not meet cross check
        matches = matches[selection]
        ds = ds[selection]

    # discard those that do not meet ratio to second-best match requirement
    if max_ratio < 1.0:
        # clip distances to very small positive number to avoid division by zero
        ds = np.clip(ds, a_min=np.finfo(float).eps, a_max=None)
        matches = matches[ds.T[1] / ds.T[0] > (1/max_ratio)]

    return matches