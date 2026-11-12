"""Multiview (multi-PSF) Richardson-Lucy deconvolution.

The same object is observed through several DIFFERENT point spread functions
(e.g. a STED microscope run with different depletion patterns, giving one
image per pattern). Each view obeys ``d_i ~ Poisson(bg_i + h_i * s)`` with one
shared latent object ``s``; the joint Poisson ML update is the multiview
Richardson-Lucy iteration

    u <- u * mean_i( flip(h_i) * [ d_i / (bg_i + h_i * u) ] )

The fused Fourier passband is the UNION of the per-view OTF supports (weighted
by SNR where they overlap), so the joint estimate can be sharper along an axis
than any single view is along that axis - but never sharper than the best view
in any given direction. Naive averaging of the views does NOT do this.
"""
import numpy as np
from scipy.ndimage import gaussian_filter

from .deconvolution import _convolve_same, _prepare_kernel


def _init_estimate(views, kernels, bg, init):
    """Initial latent estimate. init: "flat" | "bluravg" | "backproj"."""
    if init == "flat":
        u = np.full(views[0].shape, max(views[0].mean(), 1.0))
    elif init == "bluravg":
        # blurred average of the background-subtracted views: places intensity
        # at faint structures from the start (a flat init locks onto the
        # brightest structures first and can starve dim ones)
        u = np.mean([gaussian_filter(v - b, 2.0) for v, b in zip(views, bg)], axis=0)
        u = np.maximum(u, 1.0)
    elif init == "backproj":
        u = np.mean([_convolve_same(v, k[tuple(slice(None, None, -1) for _ in k.shape)])
                     for v, k in zip(views, kernels)], axis=0)
        u = np.maximum(u, 1.0)
    else:
        raise ValueError(f"unknown init: {init!r}")
    return u


def rl_deconvolve_multiview(views, kernels, *, num_iter=25, bg=0.0,
                            init="bluravg", bandlimit=None, edge="periodic",
                            damp=None, eps=1e-12):
    """Joint multiview Richardson-Lucy deconvolution (Poisson ML).

    Parameters
    ----------
    views : sequence of ndarray
        Observed images, all the same shape, one per PSF. The images are
        ASSUMED ALREADY REGISTERED to a common frame (register upstream, e.g.
        with calmutils.stitching.phase_correlation). Clipped at 0 internally.
    kernels : sequence of ndarray
        One impulse response per view, same length as ``views``. Each kernel
        is validated/normalized by :func:`_prepare_kernel` (odd shape,
        center-aligned, non-negative, unit sum).
    num_iter : int
        RL iterations. Early stopping is the implicit regularizer - on real
        photon-count data, prefer few iterations or a discrepancy-principle
        stop over many.
    bg : float or sequence of float
        Per-view background added in the RL denominator (detector offset /
        out-of-focus haze). Scalar broadcasts to all views. A wrong offset
        biases RL strongly, so estimate it per view.
    init : "bluravg" | "flat" | "backproj"
        Initialization of the latent estimate (see :func:`_init_estimate`).
        "bluravg" (default) converges fastest and preserves faint structures
        best; "flat" is the classic constant start.
    bandlimit : sequence of float, optional
        Per-axis OTF cutoff in cycles per voxel, applied after every iteration
        (same semantics as in :func:`rl_deconvolve`).
    edge : "periodic" | "reflect"
        FFT convolution is periodic. "periodic" (default) is fine for PSF
        windows that decay to ~0 at the border; "reflect" pads every view by
        the kernel half-width with mirrored content - use for full samples.
    damp : float, optional
        Denominator floor ``max(conv + bg, eps, damp)``; set relative to the
        observed peak, e.g. ``damp=1e-3 * max(view.max())``.
    eps : float
        Division guard.

    Returns
    -------
    ndarray, same shape as each view, non-negative. Not renormalized (RL
    approximately conserves total flux with unit-sum kernels).
    """
    views = [np.maximum(np.asarray(v, dtype=float), 0.0) for v in views]
    if len(views) == 0:
        raise ValueError("need at least one view")
    shape = views[0].shape
    if any(v.shape != shape for v in views):
        raise ValueError("all views must have the same shape")
    if len(kernels) != len(views):
        raise ValueError("kernels must have one entry per view")
    if np.isscalar(bg):
        bg = [float(bg)] * len(views)
    bg = [float(b) for b in bg]
    if len(bg) != len(views):
        raise ValueError("bg must be a scalar or one value per view")
    if any(v.sum() <= 0 for v in views):
        raise ValueError("each view needs some positive signal")

    kernels = [_prepare_kernel(k, len(shape)) for k in kernels]
    ndim = len(shape)

    if edge == "reflect":
        # pad per axis with the max kernel half-width across views
        pad = tuple((max(k.shape[a] for k in kernels) // 2,) * 2 for a in range(ndim))
        work = [np.pad(v, pad, mode="reflect") for v in views]
    elif edge == "periodic":
        pad = None
        work = views
    else:
        raise ValueError(f"unknown edge: {edge!r}")

    mask = None
    if bandlimit is not None:
        bl = np.atleast_1d(np.asarray(bandlimit, dtype=float))
        if len(bl) != ndim:
            raise ValueError("bandlimit must have one entry per axis")
        mask = np.ones(work[0].shape)
        for ax in range(ndim):
            f = np.fft.fftfreq(work[0].shape[ax])
            mshape = [1] * ndim
            mshape[ax] = -1
            mask *= (np.abs(f) <= bl[ax]).reshape(mshape)

    u = _init_estimate(work, kernels, bg, init)
    flip = [k[tuple(slice(None, None, -1) for _ in k.shape)] for k in kernels]
    for _ in range(int(num_iter)):
        corr = np.zeros_like(u)
        for d, h, hf, b in zip(work, kernels, flip, bg):
            conv = _convolve_same(u, h) + b
            denom = np.maximum(conv, eps)
            if damp is not None:
                denom = np.maximum(denom, float(damp))
            corr += _convolve_same(d / denom, hf)
        u = u * (corr / len(work))
        u = np.maximum(u, 0.0)
        if mask is not None:
            u = np.maximum(np.fft.ifftn(np.fft.fftn(u) * mask).real, 0.0)

    if pad is not None:
        u = u[tuple(slice(p0, p0 + n) for (p0, _), n in zip(pad, shape))]
    return u
