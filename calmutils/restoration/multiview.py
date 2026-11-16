"""Multiview (multi-PSF) Richardson-Lucy deconvolution.

The same object is observed through several DIFFERENT point spread functions
(e.g. a STED microscope run with different depletion patterns, giving one
image per pattern). Each view obeys ``d_i ~ Poisson(bg_i + a_i h_i * s)`` with
one shared latent object ``s``, unit-sum PSFs ``h_i`` and per-view relative
brightness ``a_i``; the joint Poisson ML update is the multiview
Richardson-Lucy iteration

    u <- u * sum_i( a_i H_i^T [ w_i d_i / (bg_i + a_i H_i u) ] ) / sum_i( a_i H_i^T w_i )

(H_i = convolution with h_i, H_i^T = correlation, i.e. convolution with
flip(h_i)). The a_i are part of the forward model, not ad-hoc weights: with
them in place the likelihood weights each view by the photons it carries.
The w_i are per-pixel data weights (1 = data, 0 = no data: outside the window
for padding="extend", or outside a view's valid region after registration).
For equally bright views and away from the border this reduces to the plain
mean of the single-view RL factors.

The fused Fourier passband is the UNION of the per-view OTF supports (weighted
by SNR where they overlap), so the joint estimate can be sharper along an axis
than any single view is along that axis - but never sharper than the best view
in any given direction. Naive averaging of the views does NOT do this.
"""
import numpy as np
import scipy.fft as _fft
from scipy.ndimage import gaussian_filter

from .deconvolution import (_prepare_kernel, _convolve_same_fft_shape_crop,
                            _rl_update_factor, _sensitivity,
                            _inverse_sensitivity, _prepare_weights,
                            _bandlimit_mask, _apply_bandlimit)


def rl_deconvolve_multiview(views, kernels, *, num_iter=25, bg=0.0,
                            bandlimit=None, padding=None, damp=None,
                            view_scales=None, weights=None,
                            min_sensitivity=1e-2):
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
        unit sum) - the relative brightness of the views is NOT taken from
        the kernel sums, see ``view_scales``.
    num_iter : int
        RL iterations.
    bg : float, ndarray, or sequence thereof
        Per-view background in the forward model ``y_i = H_i u + bg_i``,
        added in the RL denominator (detector offset / out-of-focus haze /
        measured bg map). Scalar broadcasts to all views; arrays must match
        the view shape. A wrong bg biases RL strongly, so estimate it per
        view.
    bandlimit : sequence of float, optional
        Per-axis OTF cutoff in cycles per voxel, applied after every iteration
        (same semantics and approximations as in :func:`rl_deconvolve`).
    padding : None | "extend"
        None (default): no padding - linear convolution with an implicit zero
        border, i.e. the object is zero outside the window; fine for objects
        that end inside the window (isolated beads, PSF crops).
        "extend": the object is estimated on a domain extended by the max
        kernel half-width across views (k//2 per side) and cropped to the
        window at the end; the margin has no data and is fitted only through
        its blur into the window (see :func:`rl_deconvolve`) - use for
        samples that continue beyond the window.
    damp : float, optional
        Denominator floor ``max(a_i * conv_i + bg_i, damp)``, on top of a tiny
        fixed division guard; set relative to the observed peak, e.g.
        ``damp=1e-3 * max(v.max() for v in views)``. A floor, not skimage's
        filter_epsilon behaviour, and it pushes regions dimmer than ``damp``
        towards zero (see :func:`rl_deconvolve`).
    view_scales : None | sequence of float
        Relative brightness a_i of each view in the forward model
        ``y_i = a_i H_i u + bg_i`` (PSF integral, laser power, dwell time,
        bleaching between acquisitions, ...). None (default): estimated from
        the data as the bg-subtracted total flux of each view (over the
        pixels where all views have data, if ``weights`` are given). A sequence
        gives one positive value per view, e.g. the sums of physically
        scaled PSFs. Either way the scales are normalized to mean 1, so only
        their ratios matter. These are model parameters, not quality weights:
        do not use them to down-weight a view by hand.
    weights : None | ndarray | sequence of (ndarray or None)
        Per-pixel data weights w_i >= 0 (0 = no data, 1 = normal), e.g. the
        valid region of each view after registration, or bad-pixel masks.
        One array of the view shape applies to all views; a sequence gives
        one entry per view (None = all ones). Where a view has weight 0 its
        values are ignored (may be NaN) and the other views determine the
        estimate. Only relative values matter.
    min_sensitivity : float
        Voxels whose summed sensitivity ``sum_i a_i H_i^T w_i`` is below this
        fraction of its maximum are set to 0 instead of being estimated (see
        :func:`rl_deconvolve`).

    Returns
    -------
    ndarray, same shape as each view, non-negative. Not renormalized: in the
    units of a view of average brightness (scale 1), with total flux
    approximately the mean bg-subtracted flux of the views.

    Notes
    -----
    Performance: same scheme as :func:`rl_deconvolve` - per-view OTFs (forward
    and flipped) are transformed once up front, FFT sizes are rounded up to
    fast lengths, the spectrum of the current estimate is shared across views,
    and transforms run through scipy.fft multithreaded. Cost per iteration is
    1 + 3*len(views) FFTs instead of 6*len(views).
    """
    
    views = [np.asarray(v, dtype=float) for v in views]
    if len(views) == 0:
        raise ValueError("need at least one view")
    shape = views[0].shape
    if any(v.shape != shape for v in views):
        raise ValueError("all views must have the same shape")
    if len(kernels) != len(views):
        raise ValueError("kernels must have one entry per view")
    nviews = len(views)
    ndim = len(shape)

    # per-view weights: one shared array or one entry per view
    if weights is None or (isinstance(weights, np.ndarray) and weights.shape == shape):
        weights = [weights] * nviews
    else:
        weights = list(weights)
        if len(weights) != nviews:
            raise ValueError("weights must be one array of the view shape or "
                             "one entry per view")
    weights = [_prepare_weights(w, shape) for w in weights]

    # zero-weight pixels carry no data; clip positive
    views = [np.maximum(v if w is None else np.where(w > 0, v, 0.0), 0.0)
             for v, w in zip(views, weights)]
    weights = [np.ones(shape) if w is None else w for w in weights]
    if any(v.sum() <= 0 for v in views):
        raise ValueError("each view needs some positive signal")

    kernels = [_prepare_kernel(k, ndim) for k in kernels]

    if padding == "extend":
        # extend per axis by the max kernel half-width across views (k//2
        # covers everything the window can see for any kernel parity)
        pad = tuple((max(k.shape[a] for k in kernels) // 2,) * 2 for a in range(ndim))
    elif padding is None:
        pad = None
    else:
        raise ValueError(f"unknown padding: {padding!r} (None or 'extend')")

    if np.ndim(bg) == 0:
        bg = [float(bg)] * nviews
    else:
        bg = list(bg)
        if len(bg) != nviews:
            raise ValueError("bg must be a scalar or one entry per view")
    norm_bg = []
    for b in bg:
        if np.ndim(b) == 0:
            norm_bg.append(float(b))
            continue
        b = np.asarray(b, dtype=float)
        if b.shape != shape:
            raise ValueError("bg arrays must match the view shape")
        norm_bg.append(b)
    bg = norm_bg

    # per-view brightness a_i (forward model y_i = a_i H_i u + bg_i),
    # normalized to mean 1
    if view_scales is None:
        # bg-subtracted total flux: sum(H_i u) = sum(u) for unit-sum kernels,
        # so the flux ratio between views is the brightness ratio - compared
        # over the pixels where all views have data
        common = np.prod(weights, axis=0)
        if not np.any(common > 0):
            raise ValueError("cannot estimate view_scales: the views share "
                             "no pixels with data; pass view_scales")
        scales = np.array([float(np.sum(common * (v - b))) for v, b in zip(views, bg)])
        if np.any(scales <= 0):
            raise ValueError("cannot estimate view_scales: a view has no "
                             "flux above its bg")
    else:
        scales = np.asarray(view_scales, dtype=float)
        if scales.shape != (nviews,):
            raise ValueError("view_scales must have one entry per view")
        if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
            raise ValueError("view_scales must be positive and finite")
    scales = scales / scales.mean()

    # init: blurred weighted average of the bg-subtracted views in object
    # units, sum_i w_i (d_i - bg_i) / sum_i w_i a_i (the estimate is bg-free)
    # - places intensity at faint structures from the start
    num = sum(w * (v - b) for w, v, b in zip(weights, views, bg))
    den = sum(a * w for a, w in zip(scales, weights))
    u = gaussian_filter(num / np.maximum(den, 1e-12 * den.max()), 2.0)
    if u.max() <= 0:
        raise ValueError("views have no signal above bg")

    # weighted data w_i * d_i and weights on the working domain; the margin
    # has weight 0, so the values of d_i and bg_i there never enter the update
    wd = [w * v for w, v in zip(weights, views)]
    if pad is not None:
        u = np.pad(u, pad, mode="reflect")  # margin start value: mirrored
        wd = [np.pad(x, pad) for x in wd]
        weights = [np.pad(w, pad) for w in weights]
        bg = [np.pad(b, pad) if np.ndim(b) > 0 else b for b in bg]
    wshape = u.shape

    # keep strictly positive (floor relative to the data scale): the RL update
    # is multiplicative, so exact zeros would stay zero forever
    u = np.maximum(u, 1e-3 * u.max())

    # FFT sizes for the linear 'same' convolutions, shared by all views: any
    # s >= work + ker - 1 per axis gives the same cropped voxels, so use the
    # max kernel extent (fast lengths come from the helper).
    kmax = tuple(max(k.shape[a] for k in kernels) for a in range(ndim))

    s, _ = _convolve_same_fft_shape_crop(wshape, kmax, ndim)
    crops, crops_adj = [], []
    for ker in kernels:
        _, crop = _convolve_same_fft_shape_crop(wshape, ker.shape, ndim)
        crops.append(crop)
        # adjoint crop offset k-1-o (o = forward crop start): exact H^T for any
        # kernel shape; equals the forward crop for odd kernels
        crops_adj.append(tuple(slice(k - 1 - c.start, k - 1 - c.start + n)
                               for c, n, k in zip(crop, wshape, ker.shape)))

    # thread pool overhead only pays off above a size threshold
    workers = -1 if u.size >= 1 << 16 else 1

    axes = tuple(range(ndim))

    # per-view OTFs computed once: forward kernel and flipped (adjoint) kernel,
    # both scaled by the view brightness a_i, so _rl_update_factor returns
    # a_i H_i^T(w_i d_i / (a_i H_i u + bg_i))
    otf = [a * _fft.rfftn(k, s=s, axes=axes, workers=workers)
           for a, k in zip(scales, kernels)]
    otf_flip = [a * _fft.rfftn(k[tuple(slice(None, None, -1) for _ in k.shape)],
                               s=s, axes=axes, workers=workers)
                for a, k in zip(scales, kernels)]

    # RL normalization 1 / sum_i a_i H_i^T w_i (the sum is nviews in the
    # interior for full weights, smaller where views have no data nearby)
    sens = np.zeros(wshape)
    for w, of, ca in zip(weights, otf_flip, crops_adj):
        sens += _sensitivity(w, of, s, ca, axes, workers)
    inv_sens = _inverse_sensitivity(sens, min_sensitivity)

    mask = (_bandlimit_mask(wshape, bandlimit, ndim)
            if bandlimit is not None else None)

    if damp is not None:
        damp = float(damp)

    for _ in range(int(num_iter)):
        U = _fft.rfftn(u, s=s, axes=axes, workers=workers)  # shared across views

        # accumulate per-view corrections / updates
        corr = np.zeros_like(u)
        for i in range(nviews):
            corr += _rl_update_factor(U, wd[i], otf[i], otf_flip[i], s,
                                      crops[i], crops_adj[i], axes, workers,
                                      damp, bg[i])

        # multiply with the summed per-view updates, normalized
        u *= corr
        u *= inv_sens
        np.maximum(u, 0.0, out=u)
        if mask is not None:
            u = _apply_bandlimit(u, mask, axes, workers)

    # crop the extended domain back to the window
    if pad is not None:
        u = u[tuple(slice(p0, p0 + n) for (p0, _), n in zip(pad, shape))]

    return u
