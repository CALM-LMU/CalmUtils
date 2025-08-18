import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
from skimage.util import label_points
from skimage.morphology import dilation
from calmutils.morphology.structuring_elements import ellipsoid_selem
from calmutils.misc.selections import safe_rectangular_selection
from calmutils.localization.gaussian_fit import gaussian_1d, gaussian_2d, gaussian_3d, gaussian_nd


def make_multi_func(func, reps=2, combine_fun=sum):
    def _multi_func(x, *params):
        res = []
        nparams_single = len(params) // reps
        for i in range(reps):
            res.append(func(x, *params[i*nparams_single:(i+1)*nparams_single]))
        return combine_fun(res)
    return _multi_func


def refine_multi_gaussian_fit(img, locations, cut_radius=5):

    # number of peaks to fit
    npeaks = len(locations)

    # get appropriate function for dimensionality
    if img.ndim == 1:
        fun = gaussian_1d
    elif img.ndim == 2:
        fun = gaussian_2d
    elif img.ndim == 3:
        fun = gaussian_3d
    else:
        fun = gaussian_nd

    # make multi function 
    fun = make_multi_func(fun, npeaks)

    # cut radius in each dimension, as integer
    if np.isscalar(cut_radius):
        cut_radius = np.full(img.ndim, cut_radius, dtype=int)
    cut_radius = np.array(cut_radius).astype(int)

    # center of locations / center of cut
    min_location = np.min(np.round(locations), axis=0)
    max_location = np.max(np.round(locations), axis=0)
    center_location = (min_location + max_location) // 2

    # cut will be from min to max plus cut_radius on each side
    cut_size = max_location - min_location + 2 * cut_radius

    # safe slices (no out-of-bounds)
    slices = safe_rectangular_selection(img, center_location, cut_size)
    
    # offset (min coords) of safe slices
    cut_offset = np.array([s.start for s in slices])

    # select cut from image
    cut = img[slices]

    # initial guess for gaussian parameters
    # for each loaction: min intensity of cut, intensity at location, location coord (in cut), cut_radius/4
    # TODO: make better estimation of sigma 
    intensity_offset = cut.min() / npeaks

    guesses = []
    for loc in locations:
        guess = [[intensity_offset, img[*loc.astype(int)]] + list(loc - cut_offset) + list(cut_radius / 4)]
        guess = np.array(guess).ravel().astype(float)
        guesses.append(guess)
    guess_multi = np.vstack(guesses)

    # mask with ellisoid around locations
    mask = label_points(locations - cut_offset, cut.shape)
    mask = dilation(mask > 0, ellipsoid_selem(cut_radius))

    # get coords in mask, select values in mask
    idxs_cut = np.argwhere(mask)
    cut = cut[mask]

    # select whole cut:
    # idxs_cut = np.stack(np.meshgrid(*(np.arange(s) for s in cut.shape), indexing='ij'), axis=-1).reshape((-1, cut.ndim)).astype(float)

    # try to optimize, return guess if optimization fails
    try:
        params_opt, _ = curve_fit(
            fun,
            idxs_cut,
            cut.ravel(),
            guess_multi
        )

    except (OptimizeWarning, RuntimeError, ValueError) as e:
        return locations, None

    # reshape flat to (npeaks, nparams)
    params_opt = params_opt.reshape(npeaks, -1)
    # add cut offset to peak locations
    params_opt[:, 2:img.ndim+2] += cut_offset
    # sum background make same for all
    params_opt[:,0] = params_opt[:,0].sum()

    return params_opt[:, 2:img.ndim+2], params_opt