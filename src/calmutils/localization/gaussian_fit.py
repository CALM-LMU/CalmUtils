import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
from numba import njit

from ..misc.selections import safe_rectangular_selection

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

    return np.array([_min] + [_max - _min] + list(com) + list(np.sqrt(var))).astype(float)

def initial_guess_gaussian2(cut):
    '''
    Faster version of guess min, max, mu_0, ..., mu_n, sigma_0, ..., sigma_n
    '''
    _min = np.min(cut)
    _max = np.max(cut)
    
    com = (np.array(np.meshgrid(*(np.arange(s) for s in cut.shape), indexing='ij')) * cut).sum(axis=tuple(range(1, cut.ndim + 1))) / cut.sum()

    var = ((np.array(np.meshgrid(*(np.arange(s) for s in cut.shape), indexing='ij')) - com.reshape((cut.ndim,) + (1,) * cut.ndim)) ** 2) * cut
    var = var.sum(axis=tuple(range(1, cut.ndim + 1))) / cut.sum()

    return np.array([_min] + [_max - _min] + list(com) + list(np.sqrt(var))).astype(float)

def gaussian_nd(x, *params):
    '''
    value of (scaled) Gaussian at the locations in x
    with parameters: min, max, mu_0, ..., mu_n, sigma_0, ..., sigma_n
    '''
    _min, _max = params[0:2]
    mu = np.array(params[2:2+int((len(params)-2)/2)], dtype=float)
    varinv = np.array(params[2+int((len(params)-2)/2):], dtype=float)**-2.0
    res = _min + _max * np.exp(-1.0/2.0 * np.dot((x - mu)**2, varinv))
    return res

'''
More explicit numba-compiled versions for 1d, 2d, 3d
'''

@njit
def gaussian_1d(x, min_, max_, mu0, sig0):
    res = min_ + max_ * np.exp(-1.0/2.0 * (((x[:,0] - mu0) / sig0)**2))
    return res

@njit
def gaussian_2d(x, min_, max_, mu0, mu1, sig0, sig1):
    res = min_ + max_ * np.exp(-1.0/2.0 * (((x[:,0] - mu0) / sig0)**2 + ((x[:,1] - mu1) / sig1)**2))
    return res

@njit
def gaussian_3d(x, min_, max_, mu0, mu1, mu2, sig0, sig1, sig2):
    res = min_ + max_ * np.exp(-1.0/2.0 * (((x[:,0] - mu0) / sig0)**2 + ((x[:,1] - mu1) / sig1)**2 + ((x[:,2] - mu2) / sig2)**2))
    return res

def refine_point_lsq(img, guess, cutregion=None, fun=None, maxmove=5):
    '''
    refine localization in img by least-squares Gaussian (cov=0) fit
    '''

    if fun is None:
        if img.ndim == 1:
            fun = gaussian_1d
        elif img.ndim == 2:
            fun = gaussian_2d
        elif img.ndim == 3:
            fun = gaussian_3d
        else:
            fun = gaussian_nd

    img_ = img
    guess_ = np.array(guess, dtype=int)

    # default cut: 5px in each direction
    if cutregion is None:
        cutregion = np.full((len(img.shape),), 5, dtype=int)

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


    slices = safe_rectangular_selection(img_, guess_, cutregion*2+1)
    # slices = tuple(slice(max(guess_[i] - cutregion[i], 0), min(guess_[i] + cutregion[i] + 1, img_.shape[i])) for i in range(len(guess)))
    cut = img_[slices]

    # initial guess for gaussian parameters
    guess_ = initial_guess_gaussian2(cut)

    # idxs_cut = np.array([idx for idx in np.ndindex(cut.shape)], dtype=float)
    # faster version without loop:
    idxs_cut = np.stack(np.meshgrid(*(np.arange(s) for s in cut.shape), indexing='ij'), axis=-1).reshape((-1, cut.ndim)).astype(float)

    # try to optimize, return guess if optimization fails
    try:
        res = curve_fit(fun,
                        idxs_cut,
                        cut.ravel(),
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
            loc, fit = refine_point_lsq(img_, (np.array([10,10]) + np.round(np.array(shift))).astype(int), [7,7])
            print((np.array([10, 10]) + np.array(shift)) - loc)

if __name__ == '__main__':
    main()