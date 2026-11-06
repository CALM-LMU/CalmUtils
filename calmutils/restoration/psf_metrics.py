import warnings

import numpy as np


def axis_profile(arr, ax):
    """Marginal along one axis: NaN-aware sum over all other axes."""
    axes = tuple(j for j in range(arr.ndim) if j != ax)
    return np.nansum(arr, axis=axes)


def profile_metrics(profile, baseline=0.0):
    """Non-parametric metrics of a 1D profile (a marginal or through-center cut).

    Parameters
    ----------
    profile : 1D array
    baseline : float
        Level subtracted before measuring (0 for background-subtracted data).
        NaN samples are treated as baseline.

    Returns
    -------
    dict with keys
      peak      : max of (profile - baseline)
      fwhm      : full width at half max in samples, subvoxel via linear
                  interpolation of the half-max crossings (NaN if peak <= 0)
      left/right: the (subvoxel) half-max crossing positions
      hw_left/hw_right : peak-to-crossing half-widths in samples
      asymmetry : (hw_right - hw_left) / (hw_right + hw_left); +1 = tail to the
                  right, -1 = tail to the left, 0 = symmetric (NaN if fwhm NaN)
      centroid  : intensity centroid in samples, within the FWHM window
      sigma     : 2nd-moment width over the whole profile (values clipped at
                  0, around the centroid)
      skew      : 3rd standardized moment over the whole profile (same clipping)
      truncated : True if the profile is still >= half max at the window edge
                  (fwhm is then only a lower bound)
    """
    p = np.asarray(profile, dtype=float)
    p = np.where(np.isfinite(p), p, baseline) - baseline
    n = p.size
    i_peak = int(np.argmax(p))
    peak = float(p[i_peak])
    out = {"peak": peak, "fwhm": np.nan, "left": np.nan, "right": np.nan,
           "hw_left": np.nan, "hw_right": np.nan, "asymmetry": np.nan,
           "centroid": np.nan, "sigma": np.nan, "skew": np.nan, "truncated": False}
    if peak <= 0 or n < 3:
        out["truncated"] = True
        return out
    h = peak / 2.0
    above = p >= h
    left = i_peak
    while left > 0 and above[left - 1]:
        left -= 1
    right = i_peak
    while right < n - 1 and above[right + 1]:
        right += 1
    out["truncated"] = bool(left == 0 or right == n - 1)
    x_l = float(np.interp(h, [p[left - 1], p[left]], [left - 1, left])) if left > 0 else 0.0
    x_r = float(np.interp(h, [p[right + 1], p[right]], [right + 1, right])) if right < n - 1 else float(n - 1)
    out["left"], out["right"] = x_l, x_r
    out["fwhm"] = x_r - x_l
    out["hw_left"] = i_peak - x_l
    out["hw_right"] = x_r - i_peak
    hw_sum = out["hw_left"] + out["hw_right"]
    if hw_sum > 0:
        out["asymmetry"] = (out["hw_right"] - out["hw_left"]) / hw_sum
    il = max(0, int(np.floor(x_l)))
    ir = min(n - 1, int(np.ceil(x_r)))
    w = p[il:ir + 1]
    idx = np.arange(il, ir + 1)
    if w.sum() > 0:
        out["centroid"] = float((idx * w).sum() / w.sum())
    pc = np.maximum(p, 0.0)
    if pc.sum() > 0 and np.isfinite(out["centroid"]):
        c = out["centroid"]
        d = np.arange(n) - c
        out["sigma"] = float(np.sqrt(((d ** 2) * pc).sum() / pc.sum()))
        if out["sigma"] > 0:
            out["skew"] = float(((d / out["sigma"]) ** 3 * pc).sum() / pc.sum())
    return out


def _edge_level(psf):
    """Max at the window border (2-voxel shell) relative to the peak -
    wraparound indicator for periodic interpolation."""
    psf = np.asarray(psf, dtype=float)
    edge = np.ones(psf.shape, dtype=bool)
    edge[tuple(slice(2, s - 2) for s in psf.shape)] = False
    if not edge.any() or not np.isfinite(psf).any() or np.nanmax(psf) <= 0:
        return 0.0
    return float(np.nanmax(psf[edge]) / np.nanmax(psf))


def _sidelobe_1d(cut):
    """Per-side sidelobe analysis of a 1D through-center cut.

    Returns dict(left, right, left_pos, right_pos): sidelobe amplitude relative
    to the central peak beyond the first local minimum on each side, and the
    distance of that maximum from the peak in samples (the STED ring radius).
    A pure Gaussian decreases monotonically -> no local minimum -> ratios ~0.
    Local-minimum detection runs on a 3-sample smoothed copy (sidelobes are
    wide; single-sample noise must not trigger)."""
    p = np.where(np.isfinite(cut), cut, 0.0)
    n = p.size
    out = {"left": 0.0, "right": 0.0, "left_pos": np.nan, "right_pos": np.nan}
    if n < 5:
        return out
    i0 = int(np.argmax(p))
    peak = float(p[i0])
    if peak <= 0:
        return out
    sm = np.convolve(p, np.ones(3) / 3.0, mode="same")
    for d, side in ((-1, "left"), (1, "right")):
        j = i0
        while 0 <= j + d < n:
            if abs(j - i0) >= 2 and sm[j + d] >= sm[j]:
                break  # local minimum at j
            j += d
        region = p[:j] if d == -1 else p[j + 1:]
        if region.size:
            k = int(np.argmax(region))
            abs_idx = k if d == -1 else j + 1 + k
            out[side] = max(float(region[k]), 0.0) / peak
            out[side + "_pos"] = float(abs(abs_idx - i0))
    return out


def psf_metrics(psf, psz=None):
    """Profile-based (non-parametric) quantification of a PSF.

    Widths come from axis marginals with subvoxel half-max interpolation -
    NOT from a Gaussian fit (STED PSFs are non-Gaussian). ``psz`` (voxel size
    per axis) additionally gives physical-unit widths and offsets.

    Returns
    -------
    flat dict:
      fwhm_px, sigma_px, centroid_px : per-axis tuples from the marginals
      peak_offset_px                 : centroid - center voxel, per axis
                                       (~0 for a centered / resampled PSF)
      sidelobe_ratio                 : worst-axis/side sidelobe of the
                                       through-center cuts: max beyond the first
                                       local minimum after the central peak,
                                       relative to the peak (STED depletion QC;
                                       ~0 for a clean Gaussian)
      sidelobe_left/right            : per-side worst-axis sidelobe ratios
      sidelobe_asym                  : (right - left)/(right + left) of the
                                       worst axis; +1 = ring only on + side
      ring_radius_px                 : per-axis distance of the stronger
                                       sidelobe maximum from the peak (STED
                                       donut radius)
      asymmetry_px                   : per-axis (hw_right - hw_left)/sum of the
                                       marginal half-widths (shape asymmetry,
                                       e.g. coma tail)
      skew                           : per-axis 3rd standardized moment of the
                                       marginals
      astigmatism                    : (fwhm_last - fwhm_second_to_last) /
                                       mean of the two (x/y FWHM split);
                                       None for 1D
      edge_level                     : window-border max / peak (wraparound
                                       indicator, same check resample_psf warns)
      truncated                      : any axis marginal truncated at the edge
      sum, min
      fwhm_phys, peak_offset_phys, ring_radius_phys : only if psz is given
    """
    psf = np.asarray(psf, dtype=float)
    ndim = psf.ndim
    center = np.array(psf.shape) // 2
    if psz is not None:
        psz = np.atleast_1d(np.asarray(psz, dtype=float))
        if len(psz) != ndim:
            raise ValueError(f"psz must have {ndim} entries to match psf dims")
    fwhm, sig, cen, asym, skew, ring = [], [], [], [], [], []
    sidelobe = 0.0
    sidelobe_l = 0.0
    sidelobe_r = 0.0
    sidelobe_asym = 0.0
    truncated = False
    for ax in range(ndim):
        m = profile_metrics(axis_profile(psf, ax))
        fwhm.append(m["fwhm"])
        sig.append(m["sigma"])
        cen.append(m["centroid"])
        asym.append(m["asymmetry"])
        skew.append(m["skew"])
        truncated |= m["truncated"]
        # through-center cut along this axis for the sidelobe analysis
        cut = psf[tuple(slice(None) if j == ax else center[j]
                        for j in range(ndim))]
        sl = _sidelobe_1d(cut)
        sidelobe = max(sidelobe, sl["left"], sl["right"])
        if sl["left"] > sidelobe_l:
            sidelobe_l = sl["left"]
        if sl["right"] > sidelobe_r:
            sidelobe_r = sl["right"]
        s = sl["left"] if sl["left"] >= sl["right"] else sl["right"]
        pos = sl["left_pos"] if sl["left"] >= sl["right"] else sl["right_pos"]
        if s > 0 and np.isfinite(pos):
            ring.append(pos)
            tot = sl["left"] + sl["right"]
            if tot > 0:
                a = (sl["right"] - sl["left"]) / tot
                if abs(a) > abs(sidelobe_asym):
                    sidelobe_asym = a
        else:
            ring.append(np.nan)
    astig = None
    if ndim >= 2 and np.all(np.isfinite(fwhm[-2:])) and np.mean(fwhm[-2:]) > 0:
        astig = float((fwhm[-1] - fwhm[-2]) / np.mean(fwhm[-2:]))
    out = {
        "fwhm_px": tuple(fwhm),
        "sigma_px": tuple(sig),
        "centroid_px": tuple(cen),
        "peak_offset_px": tuple(c - c0 for c, c0 in zip(cen, center)),
        "sidelobe_ratio": sidelobe,
        "sidelobe_left": sidelobe_l,
        "sidelobe_right": sidelobe_r,
        "sidelobe_asym": sidelobe_asym,
        "ring_radius_px": tuple(ring),
        "asymmetry_px": tuple(asym),
        "skew": tuple(skew),
        "astigmatism": astig,
        "edge_level": _edge_level(psf),
        "truncated": truncated,
        "sum": float(np.nansum(psf)),
        "min": float(np.nanmin(psf)),
    }
    if psz is not None:
        out["fwhm_phys"] = tuple(f * p for f, p in zip(fwhm, psz))
        out["peak_offset_phys"] = tuple((c - c0) * p
                                        for c, c0, p in zip(cen, center, psz))
        out["ring_radius_phys"] = tuple(r * p for r, p in zip(ring, psz))
    return out


def _radius_grid(shape, psz):
    """Physical distance of each voxel from the center voxel (sphere in
    physical space, so anisotropic psz is handled)."""
    psz = np.atleast_1d(np.asarray(psz, dtype=float))
    if len(psz) != len(shape):
        raise ValueError(f"psz must have {len(shape)} entries to match shape")
    center = np.array(shape) // 2
    grids = np.meshgrid(*[(np.arange(s) - c) * p
                          for s, c, p in zip(shape, center, psz)],
                        indexing="ij")
    return np.sqrt(sum(g * g for g in grids))


def encircled_energy(psf, radius, psz):
    """Fraction of the PSF flux within a physical ``radius`` of the center voxel.

    EE(r) = sum(psf[dist <= r]) / sum(psf), with dist in physical units (a
    sphere in physical space, not an ellipsoid in voxels). Non-parametric and
    captures sidelobe/wing energy that FWHM misses - the standard STED QC
    number ("80% of photons within 100 nm"). psf need not be unit sum.

    Warns if the PSF has not decayed at the window edge (edge_level > 0.01):
    truncation biases EE at large radii.
    """
    psf = np.asarray(psf, dtype=float)
    r = _radius_grid(psf.shape, psz)
    p = np.where(np.isfinite(psf), psf, 0.0)
    tot = p.sum()
    if tot <= 0:
        raise ValueError("psf has no positive flux")
    if _edge_level(psf) > 0.01:
        warnings.warn("psf has significant values at the window edge; "
                      "encircled energy at large radii is biased by truncation")
    return float(p[r <= radius].sum() / tot)


def radius_at_fraction(psf, q, psz):
    """Smallest physical radius containing fraction ``q`` of the PSF flux
    (inverse of :func:`encircled_energy`, subvoxel via linear interpolation
    between the sorted voxel radii)."""
    if not 0.0 < q < 1.0:
        raise ValueError("q must be in (0, 1)")
    psf = np.asarray(psf, dtype=float)
    r = _radius_grid(psf.shape, psz)
    p = np.where(np.isfinite(psf), psf, 0.0)
    tot = p.sum()
    if tot <= 0:
        raise ValueError("psf has no positive flux")
    order = np.argsort(r.ravel())
    rr = r.ravel()[order]
    cum = np.cumsum(p.ravel()[order]) / tot
    k = int(np.searchsorted(cum, q))
    if k == 0:
        return float(rr[0])
    if cum[k - 1] >= q:
        return float(rr[k - 1])
    frac = (q - cum[k - 1]) / (cum[k] - cum[k - 1])
    return float(rr[k - 1] + frac * (rr[k] - rr[k - 1]))
