from scipy import ndimage as ndi
from skimage.feature import peak_local_max
import numpy as np

from .util import sig_to_full_width_at_quantile


def detect_dog(img, threshold, sigma=None, fwhm=None, pixsize=None, steps_per_octave=4, img_sigma=None):
    """
    Difference-of-Gaussian spot detection

    Parameters
    ----------
    img: array
        image to detect peaks in
    threshold: float < 1
        intensity threshold (in normalized Dog image) for local maxima
        NB: we only detect maxima, to detect minima, run again with inverted image
    sigma: array or float
        expected sigma of spots
        if a single value is given, we use it for all dimensions
        optional, you can alternatively provide fwhm
    sigma: array or float
        expected full-width-at-half-maximum of spots
        if a single value is given, we use it for all dimensions
        optional, you can alternatively provide sigma
    pixsize: array or float
        pixel size, optional
        if not provided, we assuma sigma/fwhm in pixel units, otherwise in world units
        if a single value is given, we use it for all dimensions
    steps_per_octave: int
        number of steps per octave in a DoG-pyramid
        we use sigma1 = sigma and sigma2 = sigma * 2.0**(1/steps_per_octave) for DoG
    img_sigma: array or float
        existing blur in image, may be used to correct for anisotropy
        if a single value is given, we use it for all dimensions
        optional, if not provided we assume 0.5
        # TODO: the functionality of this parameter is a bit redundant? is this really necessary?

    Returns
    -------
    peaks: list of array
        coordinates of the detected peaks ()

    """

    # we have to provide a sigma or fwhm estimate
    if sigma is None and fwhm is None:
        raise ValueError('Please provide either sigma or fwhm estimate')
    elif sigma is None:
        fwhm = np.array(fwhm) if not np.isscalar(fwhm) else np.array([fwhm] * len(img.shape))
        sigma = fwhm / sig_to_full_width_at_quantile(np.ones_like(fwhm))
    elif fwhm is None:
        sigma = np.array(sigma) if not np.isscalar(sigma) else np.array([sigma] * len(img.shape))

    # user provided pixelsize -> assume sigma is in units
    if pixsize is not None:
        sigma /= (np.array(pixsize) if not np.isscalar(pixsize) else np.array([pixsize] * len(img.shape)))

    # image might already have a scale, assume 0.5 by default
    if img_sigma is None:
        img_sigma = np.ones_like(sigma) * 0.5
    img_sigma = np.array(img_sigma) if not np.isscalar(img_sigma) else np.array([img_sigma] * len(img.shape))
    sigma = np.sqrt(sigma ** 2 - img_sigma ** 2)

    # get DoG sigmas
    s1 = sigma
    s2 = sigma * 2.0 ** (1 / steps_per_octave)

    # do DoG, normalize result
    g1 = ndi.gaussian_filter(img, s1)
    g2 = ndi.gaussian_filter(img, s2)
    dog = g1 - g2
    dog /= np.max(dog)

    # exclude points that are closer than fwhm (in dimension with highest fwhm)
    mindist = int(np.round(np.max(sig_to_full_width_at_quantile(sigma))))
    peaks = peak_local_max(dog, min_distance=mindist, threshold_abs=threshold, exclude_border=False)

    # return as list
    return [peak for peak in list(peaks)]