"""Synthetic PSFs with known aberrations.

Ground truth for the numeric tests in ``test_restoration.py`` and for the
visual metric gallery ``psf_metrics_gallery.ipynb`` (repo root): each
aberration parameter maps to a known :func:`psf_metrics` /
:func:`second_moments` readout, so the gallery shows at a glance which metric
responds to which aberration.
"""

import numpy as np

from .shift import subpixel_shift

__all__ = ["gaussian_psf", "synthetic_psf"]


def gaussian_psf(shape, sigma=None, *, covariance=None):
    """Unit-sum multivariate Gaussian volume centered in the window.

    Parameters
    ----------
    shape : sequence of int (odd per axis for a well-defined center voxel)
    sigma : per-axis sigma in voxels (ignored if ``covariance`` given)
    covariance : full (ndim, ndim) covariance in voxels^2, overrides ``sigma``
    """
    shape = tuple(int(s) for s in shape)
    ndim = len(shape)
    if covariance is None:
        sigma = np.atleast_1d(np.asarray(sigma, dtype=float))
        if len(sigma) != ndim:
            raise ValueError(f"sigma must have {ndim} entries to match shape")
        covariance = np.diag(sigma ** 2)
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != (ndim, ndim):
        raise ValueError(f"covariance must be ({ndim}, {ndim})")
    grids = np.meshgrid(*[np.arange(s, dtype=float) - (s - 1) / 2.0
                          for s in shape], indexing="ij")
    r = np.stack(grids)  # (ndim, *shape)
    cinv = np.linalg.inv(covariance)
    q = np.einsum("i...,ij,j...->...", r, cinv, r)
    g = np.exp(-0.5 * q)
    s = g.sum()
    if s <= 0:
        raise ValueError("degenerate Gaussian (zero sum)")
    return g / s


def synthetic_psf(shape, sigma, *, orientation_deg=0.0, elongation=1.0,
                  astigmatism=0.0, coma=0.0, coma_sigma=3.0, coma_shift=4.0,
                  ring_radius=None, offset=0.0, background=0.0, noise=None,
                  seed=0):
    """Synthetic aberrated PSF with known ground truth.

    The returned array is unit-sum *before* ``background``/``noise`` are added.

    Parameters
    ----------
    shape, sigma : window size (odd per axis) and per-axis base sigma in voxels
    orientation_deg, elongation : rotate the covariance in the plane of the
        last two axes (y, x for ZYX): major principal axis at
        ``orientation_deg`` (degrees, from the first of the two axes toward the
        second; 45 = diagonal) with sigma_major = sigma_lat * elongation and
        sigma_minor = sigma_lat, where sigma_lat is the mean of the last two
        (post-astigmatism) base sigmas. Ground truth for the ``orientation_deg``
        / ``elongation`` keys of :func:`second_moments` / :func:`psf_metrics`.
    astigmatism : target value of the ``astigmatism`` key of
        :func:`psf_metrics`, i.e. (fwhm_last - fwhm_second_to_last)/mean of the
        two; the last-axis sigma is scaled so the metric reads this value.
    coma : relative amplitude of a secondary Gaussian blob on the +last-axis
        side (shifted ``coma_shift`` voxels, width ``coma_sigma``) - one-sided
        tail; ground truth for positive ``asymmetry_px`` / ``skew`` on the last
        axis.
    ring_radius : if given (voxels, in the last-two-axes plane), replaces the
        lateral profile by a pure donut ``(r^2/R^2) exp(1 - r^2/R^2)`` (zero at
        the center, ring peak at R) times the remaining-axis Gaussian. Note how
        :func:`psf_metrics` reads it: with no central peak, its sidelobe finder
        anchors on one side of the ring, so ``sidelobe_ratio`` ~ 1 and
        ``ring_radius_px`` ~ 2R (the diameter).
    offset : peak shift in voxels (scalar or per-axis), applied with a
        subvoxel Fourier shift; ground truth for ``peak_offset_px``.
    background : constant pedestal added after normalization (the 2nd-moment
        metrics subtract the border-shell median, so a pedestal should not
        distort them).
    noise : if given, Poisson noise is applied:
        ``out = poisson((psf + background) * noise) / noise``.
    seed : RNG seed for the noise.
    """
    shape = tuple(int(s) for s in shape)
    ndim = len(shape)
    sigma = np.atleast_1d(np.asarray(sigma, dtype=float)).astype(float).copy()
    if len(sigma) != ndim:
        raise ValueError(f"sigma must have {ndim} entries to match shape")
    if astigmatism != 0.0:
        # metric m = 2 (sx - sy) / (sx + sy)  ->  scale sx by sy (2+m) / (sx (2-m))
        k = sigma[-2] * (2.0 + astigmatism) / (sigma[-1] * (2.0 - astigmatism))
        sigma[-1] *= k

    cov = np.diag(sigma ** 2)
    if ndim >= 2 and (orientation_deg != 0.0 or elongation != 1.0):
        lat = 0.5 * (sigma[-2] + sigma[-1])
        th = np.radians(orientation_deg)
        rot = np.array([[np.cos(th), -np.sin(th)],
                        [np.sin(th), np.cos(th)]])
        cov[-2:, -2:] = rot @ np.diag([(lat * elongation) ** 2, lat ** 2]) @ rot.T

    if ring_radius is not None:
        # donut in the last-two-axes plane, Gaussian along the remaining axes
        R = float(ring_radius)
        if R <= 0:
            raise ValueError("ring_radius must be > 0")
        gy, gx = np.meshgrid(
            np.arange(shape[-2], dtype=float) - (shape[-2] - 1) / 2.0,
            np.arange(shape[-1], dtype=float) - (shape[-1] - 1) / 2.0,
            indexing="ij")
        r2 = gy ** 2 + gx ** 2
        lateral = (r2 / R ** 2) * np.exp(1.0 - r2 / R ** 2)  # shape shape[-2:]
        if ndim > 2:
            rest_grids = np.meshgrid(
                *[np.arange(s, dtype=float) - (s - 1) / 2.0 for s in shape[:-2]],
                indexing="ij")
            rest = np.exp(-0.5 * sum((g / s) ** 2
                                     for g, s in zip(rest_grids, sigma[:-2])))
            psf = lateral * rest.reshape(rest.shape + (1,) * 2)
        else:
            psf = lateral
        psf = psf / psf.sum()
    else:
        psf = gaussian_psf(shape, covariance=cov)

    if coma > 0:
        blob_sigma = sigma.copy()
        blob_sigma[-1] = coma_sigma
        blob = gaussian_psf(shape, covariance=np.diag(blob_sigma ** 2))
        shift = np.zeros(ndim)
        shift[-1] = coma_shift
        blob = subpixel_shift(blob, shift)
        psf = psf + coma * blob
        psf = psf / psf.sum()

    off = np.broadcast_to(np.asarray(offset, dtype=float), (ndim,))
    if np.any(off != 0):
        psf = subpixel_shift(psf, off)

    psf = psf + float(background)
    if noise is not None:
        rng = np.random.default_rng(seed)
        psf = rng.poisson(np.clip(psf, 0, None) * float(noise)) / float(noise)
    return psf
