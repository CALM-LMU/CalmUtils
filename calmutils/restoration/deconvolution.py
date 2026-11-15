import warnings

import numpy as np
import scipy.fft as _fft


def _convolve_same(img, kernel):
    """'same'-size FFT convolution, centered alignment for odd kernels."""
    if kernel.size == 1:
        return img * kernel.flat[0]
    s = [i + k - 1 for i, k in zip(img.shape, kernel.shape)]
    axes = tuple(range(img.ndim))
    F = np.fft.rfftn(img, s, axes=axes) * np.fft.rfftn(kernel, s, axes=axes)
    full = np.fft.irfftn(F, s, axes=axes)
    off = [(k - 1) // 2 for k in kernel.shape]
    return full[tuple(slice(o, o + n) for o, n in zip(off, img.shape))]


def _prepare_kernel(kernel, ndim):
    """Validate + normalize a deconvolution kernel: odd shape, matching dims,
    non-negative (clip + warn), unit sum, centered (warn if the center of mass
    is > ~0.05 px off the center voxel - recenter with subpixel_shift)."""
    ker = np.asarray(kernel, dtype=float)
    if ker.ndim != ndim:
        raise ValueError("kernel dims must match observed dims")
    if any(s % 2 == 0 for s in ker.shape):
        raise ValueError("kernel must have odd shape per axis (center voxel)")
    if np.any(ker < 0):
        warnings.warn("kernel has negative values; clipping (RL needs a "
                      "non-negative kernel)")
        ker = np.maximum(ker, 0.0)
    s = ker.sum()
    if s <= 0:
        raise ValueError("kernel has no positive mass")
    ker = ker / s
    center = np.array(ker.shape) // 2
    grids = np.meshgrid(*[np.arange(n) for n in ker.shape], indexing="ij")
    com = np.array([(g * ker).sum() for g in grids]) / s
    if np.max(np.abs(com - center)) > 0.05:
        warnings.warn(f"kernel center of mass is {np.round(com - center, 3)} px "
                      "off the center voxel; recenter with "
                      "subpixel_shift(kernel, center - com) or the result "
                      "will be shifted")
    return ker


def rl_deconvolve(observed, kernel, *, num_iter=25, bandlimit=None, edge="periodic",
                  damp=None, eps=1e-12):
    """Richardson-Lucy deconvolution with a known kernel (Poisson ML).

    Custom core (no TV regularization - see Notes). The output is NOT
    renormalized: with a unit-sum kernel RL approximately conserves total
    flux, which is what you want for sample intensities.

    Parameters
    ----------
    observed : ndarray
        Observed image (small PSF window, or a whole sample stack). Clipped
        at 0 internally (RL assumes a non-negative Poisson image).
    kernel : ndarray
        The impulse response to invert: odd-shaped, center-aligned,
        non-negative (validated by :func:`_prepare_kernel`; RL is invariant to
        its overall scale).
    num_iter : int
        RL iterations. Early stopping is the implicit regularizer.
    bandlimit : sequence of float, optional
        Per-axis OTF cutoff in CYCLES PER VOXEL (array-native - convert a
        physical cutoff nu [cycles/unit] via ``nu * psz``). Frequencies beyond
        the cutoff are suppressed after every RL iteration (bandwidth-limited
        ML, van der Voort-style): out-of-band content is noise by definition,
        and without this the division by the near-zero OTF amplifies noise
        each iteration. This - not a generic image prior like TV - is the
        principled regularizer here, because the OTF support is known optics.
        Set it AT the true cutoff, not below. None = no explicit band limit.
    edge : "periodic" | "reflect"
        The FFT convolution is periodic. "periodic" (default) is correct for
        small PSF windows that decay to ~0 at the border. "reflect" pads the
        observed image by the kernel half-width with mirrored content,
        deconvolves, and crops back - use for real samples, where periodic
        convolution would ghost border content to the opposite edge.
    damp : float, optional
        Denominator floor in the RL division (skimage's filter_epsilon-style):
        ``denom = max(conv, eps, damp)``. A cheap extra damper for near-cutoff
        noise; set relative to the observed peak, e.g. ``damp=1e-3*observed.max()``.
        None = off.
    eps : float
        Division guard.

    Returns
    -------
    ndarray, same shape as ``observed``, non-negative.

    Notes
    -----
    - No TV regularization on purpose: TV penalizes the smooth gradients that
      constitute a PSF (biases PSF distillation) and breaks intensity
      linearity (bad for quantitative sample deconvolution); bandlimit +
      num_iter + damp cover those use cases without staircasing artifacts.
    - Initialization: u starts at the observed image (skimage starts at a
      constant 0.5 - same ML fixed point, different transient; the CalmUtils
      test cross-checks against skimage).
    - Performance: the kernel OTFs (forward and adjoint) are transformed once
      (not per convolution), FFT sizes are rounded up to fast lengths, and
      transforms run through scipy.fft multithreaded - 4 FFTs per iteration
      instead of 6+.
    - The backprojection uses the ADJOINT of the forward operator, i.e.
      convolution with the flipped kernel (RL is Poisson ML / EM, and the
      update needs H^T). For symmetric kernels this is identical to the
      forward kernel; for asymmetric kernels using the unflipped kernel is
      not a descent direction and can diverge.
    """
    img = np.asarray(observed, dtype=float)
    ker = _prepare_kernel(kernel, img.ndim)
    img = np.maximum(img, 0.0)
    if img.sum() <= 0:
        raise ValueError("observed has no positive signal")

    if edge == "reflect":
        pad = tuple((k // 2, k // 2) for k in ker.shape)  # per-axis (before, after)
        work = np.pad(img, pad, mode="reflect")
    elif edge == "periodic":
        pad = None
        work = img
    else:
        raise ValueError(f"unknown edge: {edge!r}")

    ndim = img.ndim
    axes = tuple(range(ndim))
    # FFT sizes for the linear 'same' convolution. Any s >= img + ker - 1 gives
    # the same cropped voxels, so round up to a fast (smooth) length - plain
    # img + ker - 1 can contain large prime factors and be several times slower.
    s = [_fft.next_fast_len(w + k - 1, real=(ax == ndim - 1))
         for ax, (w, k) in enumerate(zip(work.shape, ker.shape))]
    crop = tuple(slice((k - 1) // 2, (k - 1) // 2 + w)
                 for w, k in zip(work.shape, ker.shape))
    # thread pool overhead only pays off above a size threshold
    workers = -1 if work.size >= 1 << 16 else 1

    otf = _fft.rfftn(ker, s=s, axes=axes, workers=workers)
    # adjoint OTF: convolution with the flipped kernel (the RL backprojection
    # needs H^T, not H; identical to `otf` for symmetric kernels)
    otf_adj = _fft.rfftn(ker[tuple(slice(None, None, -1) for _ in ker.shape)],
                         s=s, axes=axes, workers=workers)

    mask = None
    if bandlimit is not None:
        bl = np.atleast_1d(np.asarray(bandlimit, dtype=float))
        if len(bl) != ndim:
            raise ValueError("bandlimit must have one entry per axis")
        # bandlimit projection at native resolution, in the rfft half-spectrum
        # (equivalent to a full fftn * mask for real input, half the work)
        mshape = list(work.shape)
        mshape[-1] = work.shape[-1] // 2 + 1
        mask = np.ones(mshape)
        for ax in range(ndim):
            f = (_fft.rfftfreq if ax == ndim - 1 else _fft.fftfreq)(work.shape[ax])
            shape = [1] * ndim
            shape[ax] = -1
            mask *= (np.abs(f) <= bl[ax]).reshape(shape)

    if damp is not None:
        damp = float(damp)
    u = work.copy()
    for _ in range(int(num_iter)):
        U = _fft.rfftn(u, s=s, axes=axes, workers=workers)
        U *= otf
        conv = _fft.irfftn(U, s=s, axes=axes, workers=workers)[crop]
        denom = np.maximum(conv, eps)
        if damp is not None:
            denom = np.maximum(denom, damp)
        R = _fft.rfftn(work / denom, s=s, axes=axes, workers=workers)
        R *= otf_adj
        u *= _fft.irfftn(R, s=s, axes=axes, workers=workers)[crop]
        np.maximum(u, 0.0, out=u)
        if mask is not None:
            U = _fft.rfftn(u, axes=axes, workers=workers)
            U *= mask
            u = np.maximum(_fft.irfftn(U, s=work.shape, axes=axes,
                                       workers=workers), 0.0)

    if pad is not None:
        u = u[tuple(slice(p0, p0 + n) for (p0, _), n in zip(pad, img.shape))]
    return u
