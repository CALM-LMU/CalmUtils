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


def kernel_weights(points, center, bandwidth, psz=None, kind="gaussian"):
    """Kernel weights K(|p_i - center| / bandwidth) in [0, 1], max 1 at center.

    For kernel-weighted (Nadaraya-Watson style) local averaging, e.g. of PSF
    crops around a field position. Distances are physical if ``psz`` is given
    (recommended with anisotropic voxels), else in voxels; ``bandwidth`` is in
    the same unit.

    Parameters
    ----------
    points : (N, ndim) array of float
    center : (ndim,) array of float
    bandwidth : float
        Kernel width h. Rule of thumb: a few PSF FWHM, or chosen by
        leave-one-out CV. Pair with :func:`effective_sample_size` to flag
        noisy local estimates.
    psz : sequence of float, optional
    kind : "gaussian" | "epanechnikov"
        exp(-r^2/2) vs max(0, 1 - r^2).

    Returns
    -------
    (N,) ndarray of float
    """
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    c = np.asarray(center, dtype=float)
    if c.shape != (pts.shape[1],):
        raise ValueError("center must have one entry per points column")
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    if psz is None:
        scale = np.ones(pts.shape[1])
    else:
        scale = np.atleast_1d(np.asarray(psz, dtype=float))
        if len(scale) != pts.shape[1]:
            raise ValueError(f"psz must have {pts.shape[1]} entries to match points")
    r = np.linalg.norm((pts - c) * scale, axis=1) / float(bandwidth)
    if kind == "gaussian":
        return np.exp(-0.5 * r ** 2)
    if kind == "epanechnikov":
        return np.maximum(0.0, 1.0 - r ** 2)
    raise ValueError(f"unknown kernel kind: {kind!r}")


def effective_sample_size(weights):
    """Kish effective sample size (sum w)^2 / sum w^2 - how many independent
    samples a weighted average is effectively based on."""
    w = np.asarray(weights, dtype=float)
    s = w.sum()
    if s <= 0:
        return 0.0
    return float(s * s / (w ** 2).sum())
