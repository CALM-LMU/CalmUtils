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