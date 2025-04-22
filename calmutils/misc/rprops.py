def filter_rprops(rp, filt):
    """
    check if regionprops conform to a set of filters

    Parameters
    ----------
    rp: regionprop
        a single element from skimage.measure.regionprops()
    filter: dict
        map property (str) -> min, max (2-tuple)

    Return
    ------
    matches: boolean
        True if rps properties lie within all filter ranges, False otherwise

    Example
    -------
    from skimage.measure import regionprops, label
    from skimage import data
    from skimage.filters import threshold_otsu

    img = data.coins()
    filt = {
        'area': (0, 10000),
        'solidity': (0, .8)
    }

    thrd = img >= threshold_otsu(img)
    lab = label(thrd)
    rps = [rp for rp in regionprops(lab) if filter_rprops(rp, filt)]

    """
    for k, v in filt.items():
        if not (rp[k] >= v[0] and rp[k] <= v[1]):
            return False
    return True