import warnings

import numpy as np

from .psf_metrics import _edge_level


def _odd_nearest(v):
    """Nearest odd integer >= 1 to a positive value."""
    return max(1, 2 * int(v / 2.0 + 0.5) - 1)


def _resample_axis(arr, n_new, ax):
    """Resample one axis by Fourier bin remapping (band-limited, exact centering).

    Bin mapping: DC and the low positive/negative frequency bins are copied
    directly, the rest is zero (upsampling: zero-pad = sinc interpolation;
    downsampling: crop = ideal anti-aliased low-pass). Scaling by ``n_new/n``
    keeps sample *values* (PSF is a density), so the total sum changes with the
    voxel size; renormalize afterwards if a unit-sum kernel is needed.
    """
    n = arr.shape[ax]
    if n_new == n:
        return arr
    F = np.moveaxis(np.fft.fft(arr, axis=ax), ax, 0)
    G = np.zeros((n_new,) + F.shape[1:], dtype=complex)
    kmax = min((n - 1) // 2, (n_new - 1) // 2)
    G[0] = F[0]
    G[1:kmax + 1] = F[1:kmax + 1]
    G[n_new - kmax:] = F[n - kmax:]
    out = np.fft.ifft(G, axis=0).real * (n_new / n)
    return np.moveaxis(out, 0, ax)


def resample_psf(psf, psz_from, psz_to, *, out_size=None):
    """Band-limited Fourier resampling of a PSF to a new voxel size.

    Unlike spline-based rescaling (``skimage.transform.rescale``), this
    - is exact for a Nyquist-sampled PSF (a PSF is band-limited: its spectrum,
      the OTF, has finite support),
    - keeps the PSF centroid exactly on the center voxel (no half-voxel kernel
      offsets -> no deconvolution phase errors),
    - anti-aliases properly when downsampling.

    Parameters
    ----------
    psf : ndarray
        PSF with odd shape per axis (center voxel = PSF peak). Should be
        decayed to ~0 at the window edges (the interpolation is periodic);
    psz_from, psz_to : sequence of float
        Voxel size per axis, same length unit. ``psz_to`` is the *target*
        voxel size (e.g. that of the image you want to deconvolve).
    out_size : sequence of int, optional
        Output size per axis (forced odd). Default: odd size nearest to
        ``n * psz_from / psz_to``, i.e. the physical window is preserved.

    Returns
    -------
    psf_new : ndarray (unit sum), peak on the center voxel.
    psz_eff : ndarray
        The *effective* voxel size of the output, ``n * psz_from / n_new``
        per axis. Because sizes are integers (and forced odd), it matches
        ``psz_to`` only to rounding; a warning is emitted if it deviates by
        more than 5% on any axis. When matching a sample image's grid for
        deconvolution, pass ``out_size`` explicitly (an odd-sized window on
        the sample grid) and check ``psz_eff`` against the image's voxel size.
    """
    psf = np.asarray(psf, dtype=float)
    psz_from = np.atleast_1d(np.asarray(psz_from, dtype=float))
    psz_to = np.atleast_1d(np.asarray(psz_to, dtype=float))
    ndim = psf.ndim
    if len(psz_from) != ndim or len(psz_to) != ndim:
        raise ValueError("psz_from and psz_to must match psf dims")
    if any(s % 2 == 0 for s in psf.shape):
        raise ValueError("psf must have odd shape per axis (center voxel)")
    if np.any(psz_from <= 0) or np.any(psz_to <= 0):
        raise ValueError("voxel sizes must be positive")

    # warn if the PSF has not decayed at the window edge (periodic wraparound)
    if _edge_level(psf) > 0.01:
        warnings.warn("psf has significant values at the window edge; "
                      "resampling may show wraparound artifacts")

    if out_size is None:
        n_new = [_odd_nearest(n * p / q) for n, p, q in
                 zip(psf.shape, psz_from, psz_to)]
    else:
        n_new = [_odd_nearest(int(s)) for s in np.atleast_1d(out_size)]
        if len(n_new) != ndim:
            raise ValueError("out_size must have one entry per axis")
    psz_eff = np.array(psf.shape) * psz_from / np.array(n_new)
    if np.any(np.abs(psz_eff / psz_to - 1.0) > 0.05):
        warnings.warn(f"effective voxel size {psz_eff} deviates from requested "
                      f"{psz_to} by more than 5% on at least one axis; "
                      "consider passing out_size explicitly")

    # roll center voxel to origin, resample axis by axis, roll back to center
    rolled = np.roll(psf, tuple(-(s // 2) for s in psf.shape),
                     axis=tuple(range(ndim)))
    for ax in range(ndim):
        rolled = _resample_axis(rolled, n_new[ax], ax)
    out = np.roll(rolled, tuple(s // 2 for s in rolled.shape),
                  axis=tuple(range(ndim)))

    out = out / out.sum()
    return out, psz_eff
