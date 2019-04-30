import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning

def initial_guess_gaussian(cut):
    '''
    guess min, max, mu_0, ..., mu_n, sigma_0, ..., sigma_n
    '''
    _min = np.min(cut)
    _max = np.max(cut)
    com = np.array([0] * len(cut.shape), dtype=float)
    _sum = 0.0
    for idx in np.ndindex(cut.shape):
        _sum += cut[idx]
        com += np.array(idx, dtype=float) * cut[idx]
    com /= _sum

    var = np.array([0] * len(cut.shape), dtype=float)
    _sum = 0.0
    for idx in np.ndindex(cut.shape):
        _sum += cut[idx]
        var += (np.array(idx, dtype=float) - com) ** 2 * cut[idx]
    var /= _sum

    return np.array([_min] + [_max] + list(com) + list(np.sqrt(var)))


def gaussian(x, *params):
    '''
    value of (scaled) Gaussian at the locations in x
    with parameters: min, max, mu_0, ..., mu_n, sigma_0, ..., sigma_n
    '''
    _min, _max = params[0:2]
    mu = np.array(params[2:2+int((len(params)-2)/2)], dtype=float)
    varinv = np.array(params[2+int((len(params)-2)/2):], dtype=float)**-2.0
    res = _min + _max * np.exp(-1.0/2.0 * np.dot((x - mu)**2, varinv))
    return res


def refine_point_lsq(img, guess, cutregion=None, fun=gaussian, maxmove=5):
    '''
    refine localization in img by least-sqares Gaussian (cov=0) fit
    '''

    img_ = img
    guess_ = np.array(guess, dtype=np.int)

    # default cut: 5px in each direction
    if cutregion is None:
        cutregion = np.ones((len(img.shape),), dtype=np.int)*5

    # make sure cutregion is np.array
    cutregion = np.array(cutregion)

    # remember offset, in case we fit at edge of image
    off = cutregion + np.min((np.zeros(len(guess_)), guess_-cutregion), axis=0)

    # a bit overcautious:
    # pad by maximum necessary padding amount if any padding is necessary
    # this way, we always can re-index the same way
    """
    if np.any(np.greater(guess + cutregion, np.array(img.shape) - 1)) or np.any(np.less(guess - cutregion, 0)):
        guess_ = guess_ + cutregion
        img_ = np.pad(img, [(c, c) for c in cutregion], 'reflect')
    """

    # cut around blob
    idxes = [tuple(range((max(guess_[i] - cutregion[i], 0)), min(guess_[i] + cutregion[i] + 1, img_.shape[i]))) for i in range(len(guess))]
    cut = img_[np.ix_(*idxes)]

    # initial guess for gaussian parameters
    guess_ = initial_guess_gaussian(cut)

    # try to optimize, return guess if optimization fails
    try:
        res = curve_fit(fun,
                        np.array([idx for idx in np.ndindex(cut.shape)],
                                 dtype=float), cut.ravel(),
                        guess_)
    except (OptimizeWarning, RuntimeError, ValueError) as e:
        return guess, None

    # return guess if optimization deviates from guess a lot
    if np.sqrt(np.sum((guess_[2:2 + int((len(guess_) - 2) / 2)] -
                           res[0][2:2 + int((len(guess_) - 2) / 2)]) ** 2)) > maxmove:
        return guess, None

    return np.array(res[0][2:2 + int((len(guess_) - 2) / 2)], dtype=float) - off + guess, res


def main():
    from scipy import ndimage as ndi

    img = np.zeros((21,21))
    img[10,10] = 1.0

    for sigs in [(2,2), (5,5), (7,7)]:
        for shift in [[-7.8, 6.1]]:
            img_ = ndi.gaussian_filter(img, sigs)
            img_ = ndi.shift(img_, shift)
            loc, fit = refine_point_lsq(img_, (np.array([10,10]) + np.round(np.array(shift))).astype(np.int), [7,7])
            print((np.array([10, 10]) + np.array(shift)) - loc)

if __name__ == '__main__':
    main()