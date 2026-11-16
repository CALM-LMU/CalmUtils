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
import scipy.fft as _fft
from scipy.ndimage import gaussian_filter

from .deconvolution import (_prepare_kernel, _convolve_same_fft_shape_crop,
                            _rl_update_factor, _bandlimit_mask, _apply_bandlimit)


def rl_deconvolve_multiview(views, kernels, *, num_iter=25, bg=0.0,
                            bandlimit=None, padding=None, damp=None):
    """Joint multiview Richardson-Lucy deconvolution (Poisson ML).

    Parameters
    ----------
    views : sequence of ndarray
        Observed images, all the same shape, one per PSF. The images are
        ASSUMED ALREADY REGISTERED to a common frame (register upstream, e.g.
        with calmutils.stitching.phase_correlation). Clipped at 0 internally.
    kernels : sequence of ndarray
        One impulse response / PSF per view, same length as ``views``. Each kernel
        is validated/normalized by :func:`_prepare_kernel` (non-negative,
        unit sum).
    num_iter : int
        RL iterations.
    bg : float, ndarray, or sequence thereof
        Per-view background in the forward model ``y_i = H_i u + bg_i``,
        added in the RL denominator (detector offset / out-of-focus haze /
        measured bg map). Scalar broadcasts to all views; arrays must match
        the view shape (padded internally for padding="reflect"). A wrong bg
        biases RL strongly, so estimate it per view.
    bandlimit : sequence of float, optional
        Per-axis OTF cutoff in cycles per voxel, applied after every iteration
        (same semantics as in :func:`rl_deconvolve`).
    padding : None | "reflect"
        None (default): no padding - linear convolution with an implicit zero
        border; fine for PSF windows that decay to ~0 at the border.
        "reflect": pads every view by the max kernel half-width across views
        (k//2 per side) with mirrored content - use for full samples.
    damp : float, optional
        Denominator floor ``max(conv + bg, damp)``, on top of a tiny fixed
        division guard; set relative to the observed peak, e.g.
        ``damp=1e-3 * max(view.max())``.

    Returns
    -------
    ndarray, same shape as each view, non-negative. Not renormalized (RL
    approximately conserves total flux with unit-sum kernels).

    Notes
    -----
    Performance: same scheme as :func:`rl_deconvolve` - per-view OTFs (forward
    and flipped) are transformed once up front, FFT sizes are rounded up to
    fast lengths, the spectrum of the current estimate is shared across views,
    and transforms run through scipy.fft multithreaded. Cost per iteration is
    2 + 2*len(views) FFTs instead of 6*len(views).
    """
    
    # clip positive and sanity check input views
    views = [np.maximum(np.asarray(v, dtype=float), 0.0) for v in views]
    if len(views) == 0:
        raise ValueError("need at least one view")
    shape = views[0].shape
    if any(v.shape != shape for v in views):
        raise ValueError("all views must have the same shape")
    if len(kernels) != len(views):
        raise ValueError("kernels must have one entry per view")
    if any(v.sum() <= 0 for v in views):
        raise ValueError("each view needs some positive signal")

    kernels = [_prepare_kernel(k, len(shape)) for k in kernels]
    ndim = len(shape)

    if padding == "reflect":
        # pad per axis with the max kernel half-width across views (k//2 covers
        # the crop context of every view for any kernel parity)
        pad = tuple((max(k.shape[a] for k in kernels) // 2,) * 2 for a in range(ndim))
        work = [np.pad(v, pad, mode="reflect") for v in views]
    elif padding is None:
        pad = None
        work = views
    else:
        raise ValueError(f"unknown padding: {padding!r} (None or 'reflect')")

    if np.ndim(bg) == 0:
        bg = [float(bg)] * len(views)
    else:
        bg = list(bg)
        if len(bg) != len(views):
            raise ValueError("bg must be a scalar or one entry per view")
    norm_bg = []
    for b in bg:
        if np.ndim(b) == 0:
            norm_bg.append(float(b))
            continue
        b = np.asarray(b, dtype=float)
        if b.shape != shape:
            raise ValueError("bg arrays must match the view shape")
        norm_bg.append(np.pad(b, pad, mode="reflect") if pad is not None else b)
    bg = norm_bg

    # FFT sizes for the linear 'same' convolutions, shared by all views: any
    # s >= work + ker - 1 per axis gives the same cropped voxels, so use the
    # max kernel extent (fast lengths come from the helper).
    kmax = tuple(max(k.shape[a] for k in kernels) for a in range(ndim))
    
    s, _ = _convolve_same_fft_shape_crop(work[0].shape, kmax, ndim)
    crops, crops_adj = [], []
    for ker in kernels:
        _, crop = _convolve_same_fft_shape_crop(work[0].shape, ker.shape, ndim)
        crops.append(crop)
        # adjoint crop offset k-1-o (o = forward crop start): exact H^T for any
        # kernel shape; equals the forward crop for odd kernels
        crops_adj.append(tuple(slice(k - 1 - c.start, k - 1 - c.start + w)
                               for c, w, k in zip(crop, work[0].shape, ker.shape)))
    
    # thread pool overhead only pays off above a size threshold
    workers = -1 if work[0].size >= 1 << 16 else 1

    axes = tuple(range(ndim))

    # per-view OTFs computed once: forward kernel and flipped (adjoint) kernel
    otf = [_fft.rfftn(k, s=s, axes=axes, workers=workers) for k in kernels]
    otf_flip = [_fft.rfftn(k[tuple(slice(None, None, -1) for _ in k.shape)],
                           s=s, axes=axes, workers=workers) for k in kernels]

    mask = (_bandlimit_mask(work[0].shape, bandlimit, ndim)
            if bandlimit is not None else None)

    if damp is not None:
        damp = float(damp)

    # init: blurred average of the bg-subtracted views (the estimate is
    # bg-free) - places intensity at faint structures from the start
    u = np.mean([gaussian_filter(v - b, 2.0) for v, b in zip(work, bg)], axis=0)
    # keep strictly positive (floor relative to the data scale): the RL update
    # is multiplicative, so exact zeros would stay zero forever
    if u.max() <= 0:
        raise ValueError("views have no signal above bg")
    u = np.maximum(u, 1e-3 * u.max())
    nviews = len(work)
    for _ in range(int(num_iter)):
        U = _fft.rfftn(u, s=s, axes=axes, workers=workers)  # shared across views
        
        # accumulate per-view corrections / updates
        corr = np.zeros_like(u)
        for i in range(nviews):
            corr += _rl_update_factor(U, work[i], otf[i], otf_flip[i], s,
                                      crops[i], crops_adj[i], axes, workers,
                                      damp, bg[i])
        
        # multiply with averaged single-view updates
        u *= corr / nviews
        np.maximum(u, 0.0, out=u)
        if mask is not None:
            u = _apply_bandlimit(u, mask, axes, workers)

    # undo padding
    if pad is not None:
        u = u[tuple(slice(p0, p0 + n) for (p0, _), n in zip(pad, shape))]
    
    return u
