import numpy as np
from numpy.linalg import inv

try:
    import torch
except ImportError:
    torch = None

def refine_point(img, guess, maxiter=10):
    '''
    refine a point localization guess in image img by quadratic fit
    done iteratively if we jump more than half a pixel in one dim
    (SIFT subpixel localization)
    '''

    # select length-3 neighborhood of guess
    # NOTE: may select less on border of image
    idxes = tuple(slice(np.max([(g - 1), 0]), np.min([(g + 2), img.shape[i]])) for i,g in enumerate(guess))
    cut = img[idxes]

    # if image is a torch tensor, convert cut to numpy
    if torch is not None and torch.is_tensor(cut):
        cut = cut.cpu().numpy()

    # check if we are on image border and expand cut to length 3 if necessary
    min_expansion_necessary = np.equal(guess, 0)
    max_expansion_necessary = np.equal(guess, np.array(img.shape) - 1)    
    if np.any(min_expansion_necessary) or np.any(max_expansion_necessary):
        # pad with reflection -> result should be centered on border pixel and thus not move into oob
        cut = np.pad(cut, 1, 'reflect')
        idxes2 = tuple(slice(None if min_expansion_necessary[i] else 1, None if max_expansion_necessary[i] else -1) for i in range(len(guess)))
        cut = cut[idxes2]

    # get gradients, Hessian
    gr = np.gradient(cut)
    # one dimension: we need to wrap gradient in list
    if len(guess) == 1:
        gr = [gr]

    ones = tuple([1 for _ in guess])
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
    if np.any(np.abs(res) >= 0.6):
        if maxiter > 1:
            # step one pixel into direction of rounded offset
            step = np.sign(np.round(res))
            # ensure we do not step into oob, ignore move in edge cases
            for i in range(len(step)):
                if guess[i] + step[i] < 0:
                    step[i] = 0
                elif guess[i] + step[i] >= img.shape[i]:
                    step[i] = 0
            return refine_point(img, np.array(guess + step, dtype=int), maxiter - 1)
        else:
            if np.any(np.abs(res) >= 1):
                return guess
            else:
                return guess + res

    return guess + res

if __name__ == '__main__':
    from  calmutils.localization.gaussian_fit import gaussian_1d
    import numpy as np

    trace = gaussian_1d(np.arange(10, dtype=float).reshape((-1,1)), 4, 5, 2.7, 1)
    print(refine_point(trace, [2]))