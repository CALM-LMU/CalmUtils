import numpy as np


def subpixel_shift(img, shift):
    """Shift array content by ``shift`` voxels (float, one value per axis) via FFT.

    out[x] = img[x - shift], i.e. *positive* shift moves content towards higher
    indices. Band-limited (sinc) interpolation, periodic boundary.

    Note the sign convention: if an object sits at fractional offset ``off``
    relative to the crop center, pass ``-off`` here to center it.
    """
    img = np.asarray(img)
    shift = np.broadcast_to(np.asarray(shift, dtype=float), (img.ndim,))
    F = np.fft.fftn(img)
    for ax in range(img.ndim):
        f = np.fft.fftfreq(img.shape[ax])
        shape = [1] * img.ndim
        shape[ax] = -1
        F = F * np.exp(-2j * np.pi * f.reshape(shape) * shift[ax])
    return np.fft.ifftn(F).real
