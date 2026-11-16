import warnings

import numpy as np
import scipy.fft as _fft

EPS = 1e-12  # fixed numerical division guard (0/0);


def _convolve_same(img, kernel):
    """'same'-size FFT convolution;
    kernel origin at voxel (k-1)//2 per axis (left-of-center for even-shaped dims, the center voxel for odd kernels);
    the same convention as scipy.signal.convolve(mode='same')."""
    if kernel.size == 1:
        return img * kernel.flat[0]

    s, crop = _convolve_same_fft_shape_crop(img.shape, kernel.shape, img.ndim)

    axes = tuple(range(img.ndim))
    F = np.fft.rfftn(img, s, axes=axes) * np.fft.rfftn(kernel, s, axes=axes)
    full = np.fft.irfftn(F, s, axes=axes)

    return full[crop]


def _prepare_kernel(kernel, ndim):
    """Validate + normalize a deconvolution kernel: matching dims, non-negative
    (clip + warn), unit sum."""
    ker = np.asarray(kernel, dtype=float)
    if ker.ndim != ndim:
        raise ValueError("kernel dims must match observed dims")
    if np.any(ker < 0):
        warnings.warn("kernel has negative values; clipping (RL needs a "
                      "non-negative kernel)")
        ker = np.maximum(ker, 0.0)
    s = ker.sum()
    if s <= 0:
        raise ValueError("kernel has no positive mass")
    return ker / s


def _convolve_same_fft_shape_crop(image_shape, kernel_shape, ndim, even_kernel_center_convention='left'):
    """Get performant RFFTN sizes + crop (slices to extract from output)
    for 'same' convolution of image with size image_shape with kernel of kernel_shape.

    even_kernel_center_convention picks the kernel origin voxel for even k:
    'left' -> (k-1)//2 (scipy.signal.convolve 'same' convention), 'right' ->
    k//2; identical for odd k."""

    if even_kernel_center_convention not in ('left', 'right'):
        raise ValueError(f'invalid even_kernel_center_convention: {even_kernel_center_convention}. Should be "left" or "right"')
    if len(image_shape) != ndim or len(kernel_shape) != ndim:
        raise ValueError("image_shape and kernel_shape must both have length ndim")

    # FFT shape: next fast (according to backend) length s >= img + ker - 1
    fft_shape = [_fft.next_fast_len(w + k - 1, real=(ax == ndim - 1))
         for ax, (w, k) in enumerate(zip(image_shape, kernel_shape))]

    # crop from padded output: start at kernel center end at kernel center + image size
    # with even center convention "left", center is (k - 1) // 2, with "right" k // 2
    # equivalent for odd-shaped kernels
    even_off = -1 if even_kernel_center_convention == 'left' else 0
    crop = tuple(slice((k + even_off) // 2, (k + even_off) // 2 + w)
                 for w, k in zip(image_shape, kernel_shape))
    return fft_shape, crop


def _rl_update_factor(U, observed, otf, otf_adj, s, crop, crop_adj,
                      axes, workers, damp=None, bg=0.0):
    """Multiplicative correction of one RL update for a single view:
    H^T (observed / max(H u + bg, EPS, damp)), cropped to the image shape.

    U is rfftn(u, s=s, axes=axes) of the current estimate, passed in so
    multi-view deconvolution can share one transform of u across views
    (not mutated). otf/otf_adj are the view's forward/adjoint OTFs, crop/
    crop_adj the forward/adjoint 'same' crops (see
    :func:`_convolve_same_fft_shape_crop`). `observed` is the weighted data
    w * d on the working domain (zero where there is no data, e.g. in the
    margin of padding="extend"). Single-view RL applies it as
    ``u *= factor / (H^T w)``; multi-view sums the factors over views and
    divides by the summed H_i^T w_i (see :func:`_sensitivity`)."""
    conv = _fft.irfftn(U * otf, s=s, axes=axes, workers=workers)[crop]
    denom = np.maximum(conv + bg, EPS)
    if damp is not None:
        denom = np.maximum(denom, damp)
    R = _fft.rfftn(observed / denom, s=s, axes=axes, workers=workers)
    R *= otf_adj
    return _fft.irfftn(R, s=s, axes=axes, workers=workers)[crop_adj]


def _sensitivity(weights, otf_adj, s, crop_adj, axes, workers):
    """H^T w for a 'same' convolution with implicit zero border: the per-voxel
    sensitivity, i.e. how much of the light emitted at a voxel lands on
    observed pixels (weighted by w). For w = 1 and a unit-sum kernel it is 1
    in the interior and < 1 within a kernel half-width of the border; with
    w = window mask on an extended domain it falls to 0 across the margin.
    The RL update divides by it (see :func:`_inverse_sensitivity`)."""
    O = _fft.rfftn(weights, s=s, axes=axes, workers=workers)
    O *= otf_adj
    return _fft.irfftn(O, s=s, axes=axes, workers=workers)[crop_adj]


def _inverse_sensitivity(sens, min_sensitivity):
    """1 / sens where the sensitivity is above ``min_sensitivity * sens.max()``,
    0 elsewhere: voxels the data (almost) does not see are not estimated but
    set to zero (Bertero & Boccacci 2005), instead of amplifying noise by a
    huge 1 / sens."""
    valid = sens > float(min_sensitivity) * sens.max()
    return np.where(valid, 1.0 / np.where(valid, sens, 1.0), 0.0)


def _prepare_weights(weights, shape):
    """Validate a per-pixel weight map (None -> None): matching shape, finite,
    non-negative."""
    if weights is None:
        return None
    w = np.asarray(weights, dtype=float)
    if w.shape != tuple(shape):
        raise ValueError("weights must match the observed shape")
    if not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("weights must be finite and non-negative")
    if not np.any(w > 0):
        raise ValueError("weights are all zero")
    return w


def _bandlimit_mask(shape, bandlimit, ndim):
    """Float mask over the rfft half-spectrum of an array with `shape`,
    passing only frequencies within `bandlimit` (cycles per voxel, per axis).
    The passband is the product of per-axis cutoffs, i.e. a box, not an
    ellipsoid."""
    bl = np.atleast_1d(np.asarray(bandlimit, dtype=float))
    if len(bl) != ndim:
        raise ValueError("bandlimit must have one entry per axis")
    mshape = list(shape)
    mshape[-1] = shape[-1] // 2 + 1
    mask = np.ones(mshape)
    for ax in range(ndim):
        f = (_fft.rfftfreq if ax == ndim - 1 else _fft.fftfreq)(shape[ax])
        sh = [1] * ndim
        sh[ax] = -1
        mask *= (np.abs(f) <= bl[ax]).reshape(sh)
    return mask


def _apply_bandlimit(u, mask, axes, workers):
    """Project u onto the bandlimit mask (native resolution, rfft half-spectrum
    = half the work of a full fftn * mask for real input), clipped at 0.
    The transform is periodic over u (no padding), so ringing from the hard
    cutoff can wrap around to the opposite edge; the clip reintroduces a
    little out-of-band content."""
    U = _fft.rfftn(u, axes=axes, workers=workers)
    U *= mask
    return np.maximum(_fft.irfftn(U, s=u.shape, axes=axes, workers=workers), 0.0)


def rl_deconvolve(observed, kernel, *, num_iter=25, bandlimit=None, padding=None,
                  damp=None, bg=0.0, weights=None, min_sensitivity=1e-2):
    """Richardson-Lucy deconvolution with a known kernel (Poisson ML).

    The update is ``u <- u * H^T(w d / (H u + bg)) / (H^T w)`` with per-pixel
    data weights w (1 where there is data, 0 where there is none). The
    sensitivity H^T w is 1 in the interior and < 1 near the border, where part
    of the kernel falls outside the observed window; without the division,
    signal at the border is driven to zero and rings. The output is not
    renormalized: with a unit-sum kernel RL approximately conserves total
    flux (minus optional background, and minus whatever padding="extend"
    attributes to sources outside the window).

    Parameters
    ----------
    observed : ndarray
        Observed image. Clipped
        at 0 internally (RL assumes a non-negative Poisson image).
    kernel : ndarray
        The impulse response / PSF to invert: non-negative (validated by
        :func:`_prepare_kernel`; RL is invariant to its overall scale).
    num_iter : int
        RL iterations.
    bandlimit : sequence of float, optional
        Per-axis OTF cutoff in CYCLES PER VOXEL (array-native - convert a
        physical cutoff nu [cycles/unit] via ``nu * psz``). Frequencies beyond
        the cutoff are suppressed after every RL iteration (bandwidth-limited
        ML, van der Voort-style): out-of-band content is noise by definition,
        and without this the division by the near-zero OTF amplifies noise
        each iteration. This - not a generic image prior like TV - is the
        principled regularizer here, because the OTF support is known optics.
        Set it AT the true cutoff, not below. None = no explicit band limit.
        Approximations: the passband is a per-axis box (not an ellipsoid, so
        diagonal frequencies up to the corner of the box pass); the
        projection is a periodic FFT over the (padded) working array, so its
        ringing can wrap around to the opposite edge; and the result is
        clipped at 0 afterwards, which puts a little out-of-band content
        back.
    padding : None | "extend"
        None (default): no padding - FFT sizes are >= image + kernel - 1, so
        the convolution is linear with an implicit zero border (nothing wraps
        around), i.e. the object is modelled as zero outside the window. The
        sensitivity normalization keeps this stable for signal at the border,
        but light from structures just outside the window is then attributed
        to the edge voxels. Fine for objects that really end inside the
        window (isolated beads, PSF crops).
        "extend": the object is estimated on a domain extended by the kernel
        half-width (k//2 per side) and only cropped to the window at the end
        (Bertero & Boccacci 2005). The margin has no data (weight 0): it is
        fitted only through the part of its blur that lands inside the
        window, so light from outside has somewhere to go. Use for samples
        that continue beyond the window. The margin estimate starts from the
        mirrored image and is discarded.
    damp : float, optional
        Denominator floor in the RL division: ``denom = max(conv + bg, damp)``,
        on top of a tiny fixed division guard. A cheap extra damper for
        near-cutoff noise; set relative to the observed peak, e.g.
        ``damp=1e-3*observed.max()``. None = off.
        Same purpose as skimage's filter_epsilon but not the same behaviour:
        skimage sets the ratio to 0 where conv is below the threshold, this
        only floors the denominator. Side effect: where both the model and
        the observed image are below ``damp`` the ratio is < 1, so dim
        regions are pushed towards zero a little more every iteration.
    bg : float or ndarray, optional
        Background in the forward model ``y = H u + bg`` (detector offset,
        out-of-focus haze, measured bg map), added in the RL denominator.
        An array must match the observed shape. Estimate it upstream (dark frame, border
        statistics) and pass it in - the ML update wants it in the
        denominator, not pre-subtracted. Default 0.
    weights : ndarray, optional
        Per-pixel data weights w >= 0, same shape as ``observed``: 0 = no
        data at this pixel (bad pixel, saturated, outside the valid region
        after registration), 1 = normal. Zero-weight pixels are ignored
        completely (their values may be anything, including NaN) and the
        estimate there is determined by the neighbouring data through the
        kernel. Only relative values matter. None = all ones.
    min_sensitivity : float
        Voxels whose sensitivity H^T w is below this fraction of its maximum
        are set to 0 instead of being estimated (they would be amplified by
        1 / sensitivity). Only reached in the outer part of the "extend"
        margin or deep inside large zero-weight regions.

    Returns
    -------
    ndarray, same shape as ``observed``, non-negative.

    Notes
    -----
    - Initialization: u starts at the bg-subtracted observed image (skimage
      starts at a constant 0.5 - same ML fixed point, different transient).
    - Performance: the kernel OTFs (forward and adjoint) are transformed once
      (not per convolution), FFT sizes are rounded up to fast lengths, and
      transforms run through scipy.fft multithreaded - 4 FFTs per iteration
      instead of 6+.
    - The backprojection uses the ADJOINT of the forward operator: convolution
      with the flipped kernel, cropped at offset k-1-o where o is the forward
      crop offset (the two offsets coincide for odd kernels). That is the
      exact H^T for any kernel shape.
    """
    img = np.asarray(observed, dtype=float)
    ker = _prepare_kernel(kernel, img.ndim)
    w = _prepare_weights(weights, img.shape)
    if w is not None:
        img = np.where(w > 0, img, 0.0)  # zero-weight pixels carry no data
    img = np.maximum(img, 0.0)
    if img.sum() <= 0:
        raise ValueError("observed has no positive signal")

    if padding == "extend":
        # k//2 per side covers everything the window can see for any kernel
        # parity: with origin o = (k-1)//2, the conv reads k-1-o = k//2 voxels
        # left and o = (k-1)//2 <= k//2 right of each output voxel
        pad = tuple((k // 2, k // 2) for k in ker.shape)
    elif padding is None:
        pad = None
    else:
        raise ValueError(f"unknown padding: {padding!r} (None or 'extend')")

    # background is an image (e.g. measured dark image): check shape
    if np.ndim(bg) > 0:
        bg = np.asarray(bg, dtype=float)
        if bg.shape != img.shape:
            raise ValueError("bg array must match the observed shape")

    # init: bg-subtracted observed image (the estimate u is bg-free; bg lives
    # only in the denominator), mirrored into the margin
    u = np.maximum(img - bg, 0.0)
    if u.max() <= 0:
        raise ValueError("observed has no signal above bg")

    # weighted data w * d and weights on the working domain; the margin has
    # weight 0, so the values of d and bg there never enter the update
    wd = img if w is None else w * img
    if w is None:
        w = np.ones(img.shape)
    if pad is not None:
        u = np.pad(u, pad, mode="reflect")
        wd = np.pad(wd, pad)
        w = np.pad(w, pad)
        if np.ndim(bg) > 0:
            bg = np.pad(bg, pad)

    # floor at a small positive value relative to the data scale: the RL
    # update is multiplicative, so exact zeros stay zero
    u = np.maximum(u, 1e-3 * u.max())

    ndim = img.ndim
    axes = tuple(range(ndim))
    s, crop = _convolve_same_fft_shape_crop(u.shape, ker.shape, ndim)
    # adjoint crop: offset k-1-o (o = forward crop start) makes the flipped-
    # kernel convolution the exact H^T for any kernel shape; equals `crop` for odd kernels
    crop_adj = tuple(slice(k - 1 - c.start, k - 1 - c.start + n)
                     for c, n, k in zip(crop, u.shape, ker.shape))

    # thread pool overhead only pays off above a size threshold
    workers = -1 if u.size >= 1 << 16 else 1

    otf = _fft.rfftn(ker, s=s, axes=axes, workers=workers)
    # adjoint OTF: convolution with the flipped kernel (the RL backprojection
    # needs H^T, not H; identical to `otf` for symmetric kernels)
    otf_adj = _fft.rfftn(ker[tuple(slice(None, None, -1) for _ in ker.shape)],
                         s=s, axes=axes, workers=workers)

    # RL normalization 1 / (H^T w), see _sensitivity
    inv_sens = _inverse_sensitivity(
        _sensitivity(w, otf_adj, s, crop_adj, axes, workers), min_sensitivity)

    mask = (_bandlimit_mask(u.shape, bandlimit, ndim)
            if bandlimit is not None else None)

    if damp is not None:
        damp = float(damp)

    # main RL loop
    for _ in range(int(num_iter)):
        U = _fft.rfftn(u, s=s, axes=axes, workers=workers)
        u *= _rl_update_factor(U, wd, otf, otf_adj, s, crop, crop_adj,
                               axes, workers, damp, bg)
        u *= inv_sens
        np.maximum(u, 0.0, out=u)
        if mask is not None:
            u = _apply_bandlimit(u, mask, axes, workers)

    # crop the extended domain back to the window
    if pad is not None:
        u = u[tuple(slice(p0, p0 + n) for (p0, _), n in zip(pad, img.shape))]
    return u
