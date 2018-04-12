import numpy as np
from numpy.linalg import inv

def refine_point(img, guess, maxiter=10):
    '''
    refine a point localization guess in image img by quadratic fit
    done iteratively if we jump more than half a pixel in one dim
    (SIFT subpixel localization)
    '''
    ones = tuple([1 for _ in guess])

    img_ = img
    guess_ = guess

    if np.any(np.equal(guess, np.array(img.shape) - 1)) or np.any(np.equal(guess, 0)):
        guess_ = guess_ + 1
        img_ = np.pad(img, 1, 'reflect')

    idxes = [tuple(range((g - 1), (g + 2))) for g in guess_]

    cut = img_[np.ix_(*idxes)]
    gr = np.gradient(cut)
    dx = np.array([gr[i][ones] for i in range(len(guess))])

    hessian = np.zeros((len(guess), len(guess)))
    for i in range(len(guess)):
        for j in range(len(guess)):
            hessian[i, j] = np.gradient(gr[i], axis=j)[ones]

    try:
        hinv = inv(hessian)
    except np.linalg.LinAlgError:
        return guess

    res = -hinv.dot(dx) / 2
    if np.any(np.abs(res) >= 0.5):
        if maxiter > 1:
            return refine_point(img, np.array(guess + np.sign(np.round(res)) * ones, dtype=int), maxiter - 1)
        else:
            if np.any(np.abs(res) >= 1):
                return guess
            else:
                return guess + res

    return guess + res