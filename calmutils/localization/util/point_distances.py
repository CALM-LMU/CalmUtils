import numpy as np
from scipy.spatial import cKDTree


def nearest_neighbor_distances(locations, psz=None):
    """Distance of each point to its nearest *other* point.

    Parameters
    ----------
    locations : (N, ndim) array of float
        Point coordinates, e.g. subvoxel localization positions in
        array-index order.
    psz : sequence of float, optional
        Voxel size per axis. If given, coordinates are scaled by ``psz`` and
        distances are returned in physical units - recommended, because with
        anisotropic voxels (e.g. 150 nm z vs 50 nm xy) a distance in voxels
        is not meaningful. If None, distances are in voxel units.

    Returns
    -------
    (N,) ndarray of float
        Nearest-neighbour distance per point; ``inf`` if there are fewer than
        2 points. Duplicate points get distance 0.
    """
    locs = np.atleast_2d(np.asarray(locations, dtype=float))
    n = locs.shape[0]
    if n < 2:
        return np.full(n, np.inf)
    if psz is None:
        scale = np.ones(locs.shape[1])
    else:
        scale = np.atleast_1d(np.asarray(psz, dtype=float))
        if len(scale) != locs.shape[1]:
            raise ValueError(f"psz must have {locs.shape[1]} entries to match "
                             "locations columns")
    p = locs * scale
    d, _ = cKDTree(p).query(p, k=2)  # k=1 is the point itself
    return d[:, 1]
