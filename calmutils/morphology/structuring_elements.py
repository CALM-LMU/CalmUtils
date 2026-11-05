import numpy as np

def ellipsoid_selem(radii, dtype=np.uint8):
    '''
    axis-aligned ellipsoid structuring element
    dimensionality determined by length of radii iterable
    '''
    
    # NOTE: when a scalar radius is given, we default to 1D
    # TODO: this might not be terribly useful, change default or error?
    if np.isscalar(radii):
        radii = [radii]

    coords = np.stack(np.meshgrid(*[np.arange(np.floor(-r), np.ceil(r)+1) for r in radii], indexing='ij'), -1)
    selem = (coords**2 / np.array(radii)**2).sum(axis=-1) <= 1
    
    return selem.astype(dtype)

def disk_centered(radius, dtype=np.uint8):
    """
    Generates a flat, disk-shaped footprint.
    A pixel is within the neighborhood if the Euclidean distance between
    it and the origin is no greater than radius.
    Parameters
    ----------
    radius : int
        The radius of the disk-shaped footprint.
    Other Parameters
    ----------------
    dtype : data-type
        The data type of the footprint.
    Returns
    -------
    footprint : ndarray
        The footprint where elements of the neighborhood are 1 and 0 otherwise.
    """
    radius_ceil = int(np.ceil(radius))
    L = np.arange(-radius_ceil, radius_ceil + 1)
    X, Y = np.meshgrid(L, L)
    return np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)

def hypersphere_centered(rank, radius, dtype=np.uint8):
    '''
    rank-n hypersphere, should be just like disk/ball but also provide reasonable results for non-integer radii
    '''
    radius_ceil = int(np.ceil(radius))
    L = np.arange(-radius_ceil, radius_ceil + 1)
    dist = np.linalg.norm(np.stack(np.meshgrid(*[L]*rank, indexing='ij')), axis=0)
    return (dist<=radius).astype(dtype)

def bead_kernel(psz, bead_diameter, *, supersample=4):
    """
    Supersampled (anti-aliased) solid-sphere kernel, unit sum.

    Models a uniformly stained solid sphere of ``bead_diameter`` (same length
    unit as ``psz``) sampled on the voxel grid. Supersampling gives fractional
    edge voxels (correct volumes, no hard binary edge), which keeps the
    Fourier-domain conditioning much better than a binary mask - useful as a
    deconvolution kernel for PSF distillation.

    ``psz`` : one value per axis (e.g. ``(z, y, x)`` voxel size in nm). The
    kernel is an ellipsoid in voxel space (sphere in physical space) and has
    odd size ``2*ceil(radius_px) + 1`` per axis.

    Note: if the bead is smaller than a voxel, the kernel degenerates to a
    delta on the center voxel (a deconvolution with it is a no-op, as it
    should be).
    """
    psz = np.atleast_1d(np.asarray(psz, dtype=float))
    r = bead_diameter / 2.0 / psz
    if np.any(r <= 0):
        raise ValueError("bead_diameter must be positive")

    n = 2 * np.ceil(r).astype(int) + 1
    grids = np.meshgrid(
        *[(np.arange(int(s * supersample)) - (s * supersample - 1) / 2.0) / supersample
          for s in n],
        indexing="ij",
    )
    q = np.zeros(grids[0].shape)
    for g, rr in zip(grids, r):
        q += (g / rr) ** 2
    k_ss = (q <= 1.0).astype(float)

    # block-reduce the supersampled grid (mean over each supersample axis)
    newshape = [d for s in n for d in (int(s), int(supersample))]
    k = k_ss.reshape(newshape)
    k = k.mean(axis=tuple(range(1, 2 * k_ss.ndim, 2)))

    s = k.sum()
    if s <= 0:
        # bead smaller than one supersample cell: the unit-sum limit is a
        # delta on the center voxel
        k = np.zeros(tuple(int(x) for x in n))
        k[tuple(int(x) // 2 for x in n)] = 1.0
        return k
    return k / s
