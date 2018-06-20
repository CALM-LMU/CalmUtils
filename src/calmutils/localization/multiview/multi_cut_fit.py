from ..functions import GaussianNoCov
from scipy.optimize import curve_fit, OptimizeWarning
import traceback
import numpy as np

def _reduce_parameters(params, param_map):
    param_red = []
    for (i, par) in enumerate(params):
        for (j, p) in enumerate(par):
            if (i, j) in param_map:
                pass
            else:
                param_red.append(p)
    return param_red


def _expand_parameters(params, funs, param_map):
    cursor = 0
    ex_params = []
    for (i, fun) in enumerate(funs):
        params_i = []

        for j in range(0, fun.num_parameters()):
            if (i, j) in param_map:
                params_i.append(None)
            else:
                params_i.append(params[cursor])
                cursor += 1
        ex_params.append(params_i)

    for ((it, jt), (isrc, jsrc)) in param_map.items():
        ex_params[it][jt] = ex_params[isrc][jsrc]
        if ex_params[it][jt] is None:
            raise ValueError('accessing shared parameter that is itself shared.')

    return ex_params


def refine_cuts_lsq(imgs, offsets=None, pixelsize=None, fun=None, param_map=None):
    '''
    refine localization of the same point object in multiple images by least-squares function fit
    '''

    if offsets is None:
        offsets = [[0.0] * len(imgs[0].shape)] * len(imgs)

    if pixelsize is None:
        pixelsize = [[1.0] * len(imgs[0].shape)] * len(imgs)

    # default function: gaussian
    # can either be None, a single function or a list of functions
    # TODO: we only check dimensionality of first img here
    if fun is None:
        funs = [GaussianNoCov(len(imgs[i].shape), pixelsize[i], offsets[i]) for (i, _) in enumerate(imgs)]
    elif not isinstance(fun, list):
        funs = [fun] * len(imgs)
    else:
        funs = fun

    # default: no parameter sharing
    if param_map is None:
        param_map = {}

    # TODO: offsets, pixelsizes default values

    # initial guess for gaussian parameters for each image
    guesses = [funs[i].get_initial_guess(img) for (i, img) in enumerate(imgs)]

    # create parameter vector to optimize, ignore dependent instances of shared parameters
    guess_ = _reduce_parameters(guesses, param_map)

    # function to optimize, takes augmented vector (entry :,nDim determines which function we fit)
    def _fun(x, *params):

        # expand parameters so we have nFunctions separate parameter sets
        expa = _expand_parameters(params, funs, param_map)

        # split xs
        xs = []
        for i in range(len(funs)):
            xs.append(x[x[:, -1] == i, 0:-1])

        # calculate ys and re-concatenate
        res = [funs[i](xs[i], *expa[i]) for i in range(len(funs))]
        return np.concatenate(tuple(res))

    # try to optimize, return None if optimization fails
    try:
        res = curve_fit(_fun,
                        np.concatenate(tuple([np.array([list(idx) + [i] for idx in np.ndindex(img.shape)],
                                                       dtype=float) for (i, img) in enumerate(imgs)])),
                        np.concatenate(tuple([img.ravel() for img in imgs])),
                        guess_)
    except (OptimizeWarning, RuntimeError, ValueError) as e:
        traceback.print_exc()
        res = None

    return res