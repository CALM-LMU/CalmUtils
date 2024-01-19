import warnings
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage._shared.coord import ensure_spacing
import numpy as np

from .util import sigma_to_full_width_at_quantile, full_width_at_quantile_to_sigma


def detect_dog(img, threshold, sigma=None, fwhm=None, pixsize=None, sigma_factor=1.15, threshold_intensity=None,
               steps_per_octave=None, img_sigma=None, max_num_peaks=np.inf, threshold_relative=True):
    """
    Difference-of-Gaussian spot detection

    Parameters
    ----------
    img: array
        image to detect peaks in
    threshold: float < 1
        intensity threshold (in normalized DoG image) for local maxima
        NOTE: we only detect maxima. to detect minima, run again with inverted image
    sigma: array or float
        expected sigma of spots
        if a single value is given, we use it for all dimensions
        optional, you can alternatively provide fwhm
    fwhm: array or float
        expected full-width-at-half-maximum of spots
        if a single value is given, we use it for all dimensions
        optional, you can alternatively provide sigma
    pixsize: array or float
        pixel size, optional
        if not provided, we assuma sigma/fwhm in pixel units, otherwise in world units
        if a single value is given, we use it for all dimensions
    sigma_factor: float
        ratio between Gaussian blur sigmas used to compute DoG
    threshold_intensity: int/float
        intensity threshold in input image (after Gaussian blur with specified sigma)
        will be applied after maxima detection in DoG image
    steps_per_octave: int, deprecated and ignored
        number of steps per octave in a DoG-pyramid
        we use sigma1 = sigma and sigma2 = sigma * 2.0**(1/steps_per_octave) for DoG
    img_sigma: array or float, deprecated and ignored if None
        existing blur in image, may be used to correct for anisotropy
        if a single value is given, we use it for all dimensions
        optional, if not provided we assume 0.5
        # TODO: the functionality of this parameter is a bit redundant? is this really necessary?
    max_num_peaks: int
        maximum number of peaks to find in DoG response, will be enforced before filtering based on
        raw intensity or spacing, so actual number returned may be lower
    threshold_relative: bool
        whether to consider threshold relative to maximum DoG filter response

    Returns
    -------
    peaks: array of shape (n, img.ndim)
        coordinates of the n detected peaks

    """

    # warn about deprecated parameters
    if steps_per_octave is not None:
        warnings.warn('Parameter steps_per_octave is deprecated, use sigma_factor directly', DeprecationWarning)
    if img_sigma is not None:
        warnings.warn('Parameter img_sigma is deprecated, adjust sigma beforehand if desired', DeprecationWarning)

    # we have to provide a sigma or fwhm estimate
    if sigma is None and fwhm is None:
        raise ValueError('Please provide either sigma or fwhm estimate')
    elif fwhm is None:
        sigma = np.array(sigma) if not np.isscalar(sigma) else np.array([sigma] * len(img.shape))
    elif sigma is None:
        fwhm = np.array(fwhm) if not np.isscalar(fwhm) else np.array([fwhm] * len(img.shape))
        sigma = full_width_at_quantile_to_sigma(fwhm)

    # user provided pixelsize -> assume sigma is in units
    if pixsize is not None:
        sigma /= (np.array(pixsize) if not np.isscalar(pixsize) else np.array([pixsize] * len(img.shape)))

    # convert to float to avoid quantization artifacts
    img = img.astype(float)
    
    # only correct for image sigma if explicitly passed as argument
    if img_sigma is None:
        # image might already have a scale, assume 0.5 by default
        # img_sigma = np.ones_like(sigma) * 0.5
        pass
    else:
        img_sigma = np.array(img_sigma) if not np.isscalar(img_sigma) else np.array([img_sigma] * len(img.shape))
        sigma = np.sqrt(sigma ** 2 - img_sigma ** 2)

    # get DoG sigmas
    s1 = sigma / np.sqrt(sigma_factor)
    s2 = sigma * np.sqrt(sigma_factor)

    # do DoG, normalize result
    g1 = ndi.gaussian_filter(img, s1)
    g2 = ndi.gaussian_filter(img, s2)
    dog = g1 - g2

    if threshold_relative:
        dog /= np.max(dog)

    # find peaks, note: no minimal distance enforced yet
    peaks = peak_local_max(dog, min_distance=1, threshold_abs=threshold, exclude_border=False, num_peaks=max_num_peaks)

    if threshold_intensity is not None:
        # filter spots with too low raw intensity
        peaks = peaks[g1[tuple(map(list, peaks.T))] > threshold_intensity]

    # NOTE: changed to scaling with FWHM in each dimension
    # exclude points that are closer than fwhm (in dimension with highest fwhm)
    mindist = np.max(sigma_to_full_width_at_quantile(sigma))
    fwhm = sigma_to_full_width_at_quantile(sigma)
    
    # scale coords of peaks relative to maximal sigma
    rel_scale = sigma / np.max(sigma)
    peaks = peaks.astype(float) / fwhm
    # exclude closer than FWHM
    # TODO: using private function from skimage here, might break in updates
    peaks = ensure_spacing(peaks, 1, p_norm=2.0)
    # back to original coords and int
    peaks = np.round(peaks * fwhm).astype(int)

    return peaks