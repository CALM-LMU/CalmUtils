from .gaussian_fit import initial_guess_gaussian, gaussian_nd
import numpy as np

class LorentzianNoCov:
    """
    n-dimensional Lorentzian (t-dist with df=1), with diagonal shape mat
    """

    def __init__(self, num_dim, pixelsize=None, offset=None):
        self.num_dim = num_dim
        self.pixelsize = [1.0] * num_dim if pixelsize is None else pixelsize
        self.offset = [0.0] * num_dim if offset is None else offset

    def __call__(self, x, *params):
        _x = x * np.array(self.pixelsize) + np.array(self.offset)
        return self._lorentzian(_x, self.num_dim, *params)

    def num_parameters(self):
        # min, max, mus, fwhms
        return 2 * (1 + self.num_dim)

    def get_initial_guess(self, img):
        # we use the gaussian estimate
        guess = initial_guess_gaussian(img)

        # means -> to offsetted and units
        guess[2:2 + self.num_dim] *= np.array(self.pixelsize)
        guess[2:2 + self.num_dim] += np.array(self.offset)

        # sigmas -> to units
        guess[2 + self.num_dim:] *= np.array(self.pixelsize)
        # Lorentzian needs FWHMS -> Gaussian sigmas to FWHM
        # FIXME: do an actual estimate
        guess[2 + self.num_dim:] *= 2.0 * np.sqrt(2.0 * np.log(2))

        return guess

    @staticmethod
    def _lorentzian(x, num_dim, *params):
        '''
        value of (scaled) Lorentzian at the locations in x
        with parameters: min, max, mu_0, ..., mu_n, fwhm_0, ..., fwhm_n
        '''
        _min, _max = params[0:2]
        mu = np.array(params[2:2 + int((len(params) - 2) / 2)], dtype=float)
        # parameter is FWHM, inverse scale mat is HalfWidth^-2
        scaleinv = (np.array(params[2 + int((len(params) - 2) / 2):], dtype=float) / 2) ** -2.0
        l = 1.0 / (1.0 + np.dot((x - mu) ** 2, scaleinv)) ** ((1.0 + num_dim) / 2.0)

        res = _min + _max * l
        return res


class GaussianNoCov:
    """
    n-dimensional Gaussian with diagonal covariance mat (can be anisotropic, but not rotated)
    """

    def __init__(self, num_dim, pixelsize=None, offset=None):
        self.num_dim = num_dim
        self.pixelsize = [1.0] * num_dim if pixelsize is None else pixelsize
        self.offset = [0.0] * num_dim if offset is None else offset

    def __call__(self, x, *params):
        _x = x * np.array(self.pixelsize) + np.array(self.offset)
        return gaussian_nd(_x, *params)

    def num_parameters(self):
        # min, max, mus, sigs
        return 2 * (1 + self.num_dim)

    def get_initial_guess(self, img):
        guess = initial_guess_gaussian(img)

        # means -> to offsetted and units
        guess[2:2 + self.num_dim] *= np.array(self.pixelsize)
        guess[2:2 + self.num_dim] += np.array(self.offset)

        # sigmas -> to units
        guess[2 + self.num_dim:] *= np.array(self.pixelsize)

        return guess