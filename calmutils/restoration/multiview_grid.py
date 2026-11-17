"""Multiview Richardson-Lucy deconvolution for views sampled on DIFFERENT grids.

Views with different PSF / resolution trade-offs are best acquired with
different sampling (e.g. coarse plane spacing for a view with poor axial
resolution). Instead of resampling the views to a common grid (which
over-weights upsampled views in the likelihood and mis-models
undersampling), the sampling enters the forward model:

    d_i ~ Poisson(a_i S_i H_i u + bg_i)

with the latent object u on one fine object grid, H_i the view's PSF on that
grid and S_i point sampling at the view's sample positions. All geometry is
axis-aligned and per axis:

- object grid: x = origin + k * spacing
- view i:      x = origin_i + n * spacing_i     (origin = position of the
  CENTRE of voxel 0, in physical units, INCLUDING the registration shift)

Each view step must be an integer multiple r_i of the object spacing. The
offset of sample 0 in object voxels splits into an integer part m_i and a
fraction f_i in [-1/2, 1/2): sample n reads (h_i * u)(m_i + n r_i + f_i).
The integer part and the step become a slice (the view is embedded on the
object grid as a comb, with comb-shaped data weights - exactly S_i^T for
point sampling), the fraction is absorbed into the kernel,
h_i'(k) = h_i(k + f_i), since translation commutes with convolution. The
joint update is then the plain multiview RL of
:func:`rl_deconvolve_multiview`, run on the object grid.

Conventions that matter:

- Translation must be represented ONCE: kernels carry no net translation
  (centroid on the centre voxel, e.g. bead PSFs centred on their sub-voxel
  centroid), all translation is in the origins.
- Sampling is modelled as point sampling. Continuous scanning (counts binned
  over the dwell time while the beam moves) is a box along the fast axis;
  kernels measured from beads with the same scan settings contain it
  already, model PSFs need it convolved in.
"""
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .deconvolution import RLInfo, _prepare_weights
from .multiview import rl_deconvolve_multiview
from .resampling import _resample_axis
from .shift import subpixel_shift

# max accumulated position error over a view (in object voxels) from treating
# a nearly-integer step ratio as integer
_MAX_DRIFT = 0.1
# fractional offsets / index tolerances below this (object voxels) count as 0
_TOL = 1e-6


@dataclass(frozen=True)
class ObjectGrid:
    """Axis-aligned object grid: voxel k sits at ``origin + k * spacing``
    (physical units, per axis)."""
    origin: np.ndarray
    spacing: np.ndarray
    shape: tuple

    @property
    def ndim(self):
        return len(self.shape)

    def coords(self, axis):
        """Physical coordinates of the voxel centres along one axis."""
        return self.origin[axis] + np.arange(self.shape[axis]) * self.spacing[axis]

    def to_index(self, positions):
        """Physical positions (..., ndim) -> fractional voxel indices."""
        return (np.asarray(positions, dtype=float) - self.origin) / self.spacing

    def to_physical(self, indices):
        """Voxel indices (..., ndim) -> physical positions."""
        return self.origin + np.asarray(indices, dtype=float) * self.spacing


@dataclass(frozen=True)
class ViewPlacement:
    """Where a view's samples land on the object grid: ``obj[object_slices]
    = view[view_slices]``; ``frac`` is the sub-voxel offset (object voxels,
    per axis, in [-1/2, 1/2)) absorbed into the kernel; ``steps`` the integer
    step ratios r_i."""
    view_slices: tuple
    object_slices: tuple
    steps: tuple
    frac: np.ndarray


@dataclass
class GridRLInfo:
    """Diagnostics of :func:`rl_deconvolve_multiview_grid`.

    Attributes
    ----------
    rl : RLInfo
        Diagnostics of the core run on the object grid (iterations, raw
        per-view sensitivity a_i H_i^T S_i^T w_i, estimated mask, the
        normalized view_scales used).
    coverage : ndarray, shape (nviews, *grid.shape)
        Per-view sensitivity normalized to ~1 where the view is fully
        present with weight 1 (divided by a_i / prod(steps_i)), 0 where it
        has no data within a kernel half-width. ``(coverage > 0.5).sum(0)``
        counts the views constraining each voxel; ripple with the view's
        step inside its footprint means the view is undersampled for its
        kernel along that axis.
    placements : list of ViewPlacement
    kernels : list of ndarray
        The kernels as used, on the object grid (resampled, sub-voxel shifted).
    """
    rl: RLInfo
    coverage: np.ndarray
    placements: list
    kernels: list


def _per_view_axes(x, nviews, ndim, name):
    a = np.asarray(x, dtype=float)
    if a.shape != (nviews, ndim):
        raise ValueError(f"{name} must have one length-{ndim} entry per view "
                         f"(shape ({nviews}, {ndim})), got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite")
    return a


def _per_view(x, nviews, name):
    """None / scalar -> repeated; sequence -> list of length nviews (no
    np.ndim on the whole: a list of differently shaped arrays is ragged)."""
    if x is None or np.isscalar(x) or (isinstance(x, np.ndarray) and x.ndim == 0):
        return [x] * nviews
    x = list(x)
    if len(x) != nviews:
        raise ValueError(f"{name} must be a scalar/None or one entry per view")
    return x


def object_grid(shapes, spacings, origins, *, spacing=None, extent="union",
                ref=None):
    """Object grid for views on different (axis-aligned) grids.

    Parameters
    ----------
    shapes : sequence of tuple
        Shape of each view.
    spacings, origins : array-like, shape (nviews, ndim)
        Sample step and position of the centre of voxel 0 of each view,
        physical units (origins include the registration shift).
    spacing : float or sequence, optional
        Object grid spacing per axis. Default: the finest view step per axis.
        Every view step must be an integer multiple of it (checked when the
        views are placed).
    extent : "union" | "intersection" | (lo, hi)
        Region to estimate, as positions of outermost voxel centres:
        all views' sample extents (snapped outward), their common part
        (snapped inward), or an explicit physical box (snapped outward).
    ref : int, optional
        View the grid is aligned to (its samples sit exactly on object
        voxels, so its kernel is not shifted). Default: the view that is
        finest on the most axes (first one on ties).

    Returns
    -------
    ObjectGrid
    """
    shapes = np.asarray(shapes, dtype=int)
    if shapes.ndim != 2 or len(shapes) == 0:
        raise ValueError("shapes must be one shape tuple per view")
    nviews, ndim = shapes.shape
    spacings = _per_view_axes(spacings, nviews, ndim, "spacings")
    origins = _per_view_axes(origins, nviews, ndim, "origins")
    if np.any(spacings <= 0):
        raise ValueError("spacings must be positive")
    if np.any(shapes < 1):
        raise ValueError("view shapes must be positive")

    if spacing is None:
        spacing = spacings.min(axis=0)
    else:
        spacing = np.broadcast_to(np.asarray(spacing, dtype=float), (ndim,)).copy()
        if np.any(spacing <= 0) or not np.all(np.isfinite(spacing)):
            raise ValueError("spacing must be positive and finite")

    if ref is None:
        ref = int(np.argmax(np.isclose(spacings, spacing, rtol=1e-6).sum(axis=1)))
    elif not 0 <= int(ref) < nviews:
        raise ValueError(f"ref must be a view index in [0, {nviews})")
    o_ref = origins[int(ref)]

    lo_i = origins
    hi_i = origins + (shapes - 1) * spacings
    if isinstance(extent, str):
        if extent == "union":
            lo, hi, outward = lo_i.min(axis=0), hi_i.max(axis=0), True
        elif extent == "intersection":
            lo, hi, outward = lo_i.max(axis=0), hi_i.min(axis=0), False
            if np.any(hi < lo):
                raise ValueError("the views do not overlap (empty intersection)")
        else:
            raise ValueError(f"unknown extent: {extent!r} "
                             "('union', 'intersection' or (lo, hi))")
    else:
        lo, hi = (np.asarray(e, dtype=float) for e in extent)
        if lo.shape != (ndim,) or hi.shape != (ndim,) or np.any(hi < lo):
            raise ValueError("extent box must be (lo, hi) with lo <= hi, "
                             "one value per axis each")
        outward = True

    a = (lo - o_ref) / spacing
    b = (hi - o_ref) / spacing
    if outward:
        k_lo, k_hi = np.floor(a + _TOL), np.ceil(b - _TOL)
    else:
        k_lo, k_hi = np.ceil(a - _TOL), np.floor(b + _TOL)
    shape = tuple(int(n) for n in k_hi - k_lo + 1)
    if any(n < 1 for n in shape):
        raise ValueError("the extent contains no object voxel")
    return ObjectGrid(origin=o_ref + k_lo * spacing, spacing=spacing,
                      shape=shape)


def _place_axis(n_view, step_view, origin_view, o, d, n_obj):
    """One axis of :func:`place_view`: (view slice, object slice, step, frac),
    slices None if no sample of the view falls inside the object extent."""
    r = step_view / d
    r_int = max(1, int(round(r)))
    if abs(r - r_int) * max(n_view - 1, 1) > _MAX_DRIFT:
        raise ValueError(
            f"view step {step_view} is not an integer multiple of the object "
            f"spacing {d} (ratio {r:.6g}); choose an object spacing that "
            "divides all view steps (non-integer ratios are not supported)")
    kappa = (origin_view - o) / d
    m = int(np.floor(kappa + 0.5))
    f = kappa - m
    if abs(f) < _TOL:
        f = 0.0
    # samples with 0 <= m + n * r <= n_obj - 1
    n0 = max(0, -(m // r_int))                 # ceil(-m / r)
    n1 = min(n_view, (n_obj - 1 - m) // r_int + 1)
    if n1 <= n0:
        return None, None, r_int, f
    return (slice(n0, n1),
            slice(m + n0 * r_int, m + (n1 - 1) * r_int + 1, r_int),
            r_int, f)


def place_view(view_shape, view_spacing, view_origin, grid):
    """Placement of a view's samples on the object grid (see module doc).

    Returns a :class:`ViewPlacement`, or None if no sample of the view falls
    inside the grid's extent. Raises if a view step is not an integer
    multiple of the object spacing."""
    per_axis = [_place_axis(int(n), float(s), float(o), float(go), float(gd), int(gn))
                for n, s, o, go, gd, gn in zip(view_shape, view_spacing,
                                               view_origin, grid.origin,
                                               grid.spacing, grid.shape)]
    if any(p[0] is None for p in per_axis):
        return None
    return ViewPlacement(view_slices=tuple(p[0] for p in per_axis),
                         object_slices=tuple(p[1] for p in per_axis),
                         steps=tuple(p[2] for p in per_axis),
                         frac=np.array([p[3] for p in per_axis]))


def kernel_on_object_grid(kernel, kernel_spacing, spacing, frac=None):
    """Kernel resampled to the object grid and shifted by a sub-voxel offset:
    ``h'(k) = h((k + frac) * spacing)``, unit sum.

    Band-limited (Fourier) resampling as in :func:`resample_psf`, but to the
    exact size ``n * kernel_spacing / spacing`` (which must be an integer,
    e.g. any integer upsampling ratio), keeping the centre voxel centred; an
    even result is cropped by one voxel to stay odd. The shift is a Fourier
    shift on a slightly padded kernel. The kernel must be odd-shaped with its
    centroid on the centre voxel, and decayed to ~0 at its edges.
    """
    ker = np.asarray(kernel, dtype=float)
    ndim = ker.ndim
    if any(n % 2 == 0 for n in ker.shape):
        raise ValueError("kernel must have odd shape per axis (centroid on the "
                         "centre voxel)")
    kernel_spacing = np.broadcast_to(np.asarray(kernel_spacing, dtype=float), (ndim,))
    spacing = np.broadcast_to(np.asarray(spacing, dtype=float), (ndim,))
    n_new_f = np.array(ker.shape) * kernel_spacing / spacing
    n_new = np.rint(n_new_f).astype(int)
    if np.any(np.abs(n_new_f - n_new) > 1e-3) or np.any(n_new < 1):
        raise ValueError(
            f"kernel of shape {ker.shape} at spacing {tuple(kernel_spacing)} "
            f"does not map to a whole number of voxels at spacing "
            f"{tuple(spacing)} ({tuple(np.round(n_new_f, 3))}); resample it "
            "with resample_psf or crop it first")

    axes = tuple(range(ndim))
    out = np.roll(ker, [-(n // 2) for n in ker.shape], axis=axes)
    for ax, n in enumerate(n_new):
        out = _resample_axis(out, int(n), ax)
    out = np.roll(out, [n // 2 for n in out.shape], axis=axes)
    # even length: the centre sits at n // 2, drop voxel 0 to centre it in
    # an odd window
    out = out[tuple(slice(1, None) if n % 2 == 0 else slice(None)
                    for n in out.shape)]

    if frac is not None:
        frac = np.broadcast_to(np.asarray(frac, dtype=float), (ndim,))
        if np.any(frac != 0):
            out = np.pad(out, 2)  # room for the periodic shift
            out = subpixel_shift(out, -frac)  # out[k] = in[k + frac]

    out = np.maximum(out, 0.0)
    s = out.sum()
    if s <= 0:
        raise ValueError("kernel has no positive mass")
    return out / s


def _interp_matrix_1d(pos, n):
    """Sparse (len(pos), n) linear interpolation matrix; rows of positions
    outside [0, n - 1] are empty."""
    pos = np.asarray(pos, dtype=float)
    inside = (pos >= -_TOL) & (pos <= n - 1 + _TOL)
    rows = np.nonzero(inside)[0]
    p = np.clip(pos[rows], 0, n - 1)
    i0 = np.minimum(np.floor(p).astype(int), max(n - 2, 0))
    t = p - i0
    if n == 1:
        r, c, v = rows, i0, np.ones_like(t)
    else:
        r = np.concatenate([rows, rows])
        c = np.concatenate([i0, i0 + 1])
        v = np.concatenate([1 - t, t])
    return sparse.csr_matrix((v, (r, c)), shape=(len(pos), n))


def _to_object_grid_linear(arr, view_spacing, view_origin, grid):
    """Separable linear interpolation of a view onto the object grid (0
    outside the view's sample extent)."""
    out = np.asarray(arr, dtype=float)
    for ax in range(out.ndim):
        pos = (grid.coords(ax) - view_origin[ax]) / view_spacing[ax]
        M = _interp_matrix_1d(pos, out.shape[ax])
        moved = np.moveaxis(out, ax, 0)
        res = M @ moved.reshape(moved.shape[0], -1)
        out = np.moveaxis(res.reshape((M.shape[0],) + moved.shape[1:]), 0, ax)
    return out


def _estimate_view_scales(views, weights, bgs, spacings, origins):
    """Relative brightness a_i per view, as the mean bg-subtracted signal PER
    SAMPLE over the region all views cover (total flux scales with the
    number of samples, so it is not comparable across samplings)."""
    shapes = np.array([v.shape for v in views])
    lo = origins.max(axis=0)
    hi = (origins + (shapes - 1) * spacings).min(axis=0)
    if np.any(hi < lo):
        raise ValueError("cannot estimate view_scales: the views share no "
                         "footprint; pass view_scales")
    scales = []
    for i, (v, w, b, s, o) in enumerate(zip(views, weights, bgs, spacings, origins)):
        i_lo = np.ceil((lo - o) / s - _TOL).astype(int)
        i_hi = np.floor((hi - o) / s + _TOL).astype(int)
        if np.any(i_hi < i_lo):
            raise ValueError(f"cannot estimate view_scales: view {i} has no "
                             "sample in the shared footprint; pass view_scales")
        sl = tuple(slice(a, c + 1) for a, c in zip(i_lo, i_hi))
        ww = w[sl]
        if ww.sum() <= 0:
            raise ValueError(f"cannot estimate view_scales: view {i} has no "
                             "weight in the shared footprint; pass view_scales")
        bb = b[sl] if np.ndim(b) > 0 else b
        scales.append(float(np.sum(ww * (v[sl] - bb)) / ww.sum()))
    scales = np.array(scales)
    if np.any(scales <= 0):
        raise ValueError("cannot estimate view_scales: a view has no signal "
                         "above bg in the shared footprint")
    return scales


def _init_estimate(views, weights, bgs, scales, spacings, origins, grid):
    """Start estimate on the object grid: linearly interpolated, weighted
    average of the bg-subtracted views in object units,
    sum_i w_i (d_i - bg_i) / sum_i a_i w_i (no comb structure)."""
    num = np.zeros(grid.shape)
    den = np.zeros(grid.shape)
    for v, w, b, a, s, o in zip(views, weights, bgs, scales, spacings, origins):
        num += _to_object_grid_linear(w * (v - b), s, o, grid)
        den += a * _to_object_grid_linear(w, s, o, grid)
    valid = den > 1e-6 * den.max()
    u = np.where(valid, num / np.where(valid, den, 1.0), 0.0)
    u = np.maximum(u, 0.0)
    if u.max() <= 0:
        raise ValueError("views have no signal above bg")
    return u


def rl_deconvolve_multiview_grid(views, kernels, spacings, origins, *,
                                 spacing=None, extent="union", ref=None,
                                 kernel_spacings=None, weights=None, bg=0.0,
                                 view_scales=None, init=None,
                                 return_info=False, **rl_kwargs):
    """Joint multiview RL deconvolution of views sampled on different grids.

    See the module docstring for the model and conventions. The views are
    placed on a common object grid by position and step (axis-aligned;
    each view step an integer multiple of the object spacing), embedded as
    combs with comb-shaped weights, and deconvolved jointly with
    :func:`rl_deconvolve_multiview`.

    Parameters
    ----------
    views : sequence of ndarray
        Observed images, one per PSF, any shapes (same ndim).
    kernels : sequence of ndarray
        One PSF per view, odd-shaped, centroid on the centre voxel (no net
        translation), sampled at ``kernel_spacings``.
    spacings, origins : array-like, shape (nviews, ndim)
        Per view and axis: sample step and position of the centre of voxel
        0, physical units. Origins include the registration shift.
    spacing, extent, ref
        Object grid, see :func:`object_grid`.
    kernel_spacings : array-like, shape (nviews, ndim), optional
        Sampling of the kernels if not that of their views (e.g. a PSF
        measured at a different step, or already given on the object grid).
    weights : None | sequence of (ndarray or None)
        Per-view, per-pixel data weights in the view's own shape (0 = no
        data; the value there is ignored and may be NaN).
    bg : float, ndarray, or sequence thereof
        Per-view background (scalar or view-shaped array).
    view_scales : None | sequence of float
        Per-view brightness a_i, per SAMPLE. None: estimated as the mean
        bg-subtracted signal per sample over the footprint all views share.
    init : ndarray, optional
        Start estimate on the object grid (``grid.shape``). None: the views
        linearly interpolated onto the object grid and averaged (weighted,
        in object units).
    return_info : bool
        If True, also return a :class:`GridRLInfo`.
    **rl_kwargs
        Passed to :func:`rl_deconvolve_multiview` (num_iter, padding,
        bandlimit (cycles per OBJECT voxel), damp, min_sensitivity,
        callback - which sees the estimate on the object grid).

    Returns
    -------
    u : ndarray
        Estimate on the object grid, in the units of a view of average
        per-sample brightness (scale 1).
    grid : ObjectGrid
        Origin, spacing and shape of ``u``.
    info : GridRLInfo
        Only with ``return_info=True``.
    """
    views = [np.asarray(v, dtype=float) for v in views]
    nviews = len(views)
    if nviews == 0:
        raise ValueError("need at least one view")
    ndim = views[0].ndim
    if any(v.ndim != ndim for v in views):
        raise ValueError("all views must have the same number of dimensions")
    if len(kernels) != nviews:
        raise ValueError("kernels must have one entry per view")
    spacings = _per_view_axes(spacings, nviews, ndim, "spacings")
    origins = _per_view_axes(origins, nviews, ndim, "origins")
    kernel_spacings = (spacings if kernel_spacings is None else
                       _per_view_axes(kernel_spacings, nviews, ndim,
                                      "kernel_spacings"))

    if weights is not None and isinstance(weights, np.ndarray):
        raise ValueError("weights must be one entry per view (views can "
                         "differ in shape)")
    weights = _per_view(weights, nviews, "weights")
    weights = [np.ones(v.shape) if w is None else _prepare_weights(w, v.shape)
               for v, w in zip(views, weights)]
    # zero-weight pixels carry no data (may be NaN)
    views = [np.where(w > 0, v, 0.0) for v, w in zip(views, weights)]

    bgs = []
    for v, b in zip(views, _per_view(bg, nviews, "bg")):
        if np.ndim(b) == 0:
            bgs.append(float(b))
        else:
            b = np.asarray(b, dtype=float)
            if b.shape != v.shape:
                raise ValueError("bg arrays must match their view's shape")
            bgs.append(b)

    grid = object_grid([v.shape for v in views], spacings, origins,
                       spacing=spacing, extent=extent, ref=ref)
    placements = []
    for i, (v, s, o) in enumerate(zip(views, spacings, origins)):
        p = place_view(v.shape, s, o, grid)
        if p is None:
            raise ValueError(f"view {i} lies entirely outside the object extent")
        placements.append(p)

    if view_scales is None:
        scales = _estimate_view_scales(views, weights, bgs, spacings, origins)
    else:
        scales = np.asarray(view_scales, dtype=float)
        if scales.shape != (nviews,):
            raise ValueError("view_scales must have one entry per view")
        if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
            raise ValueError("view_scales must be positive and finite")
    scales = scales / scales.mean()

    kernels_o = [kernel_on_object_grid(k, ks, grid.spacing, p.frac)
                 for k, ks, p in zip(kernels, kernel_spacings, placements)]

    # comb embedding: S_i^T d_i, S_i^T w_i (and bg maps on the same samples)
    def embed(arr, p):
        out = np.zeros(grid.shape)
        out[p.object_slices] = arr[p.view_slices]
        return out

    views_o = [embed(v, p) for v, p in zip(views, placements)]
    weights_o = [embed(w, p) for w, p in zip(weights, placements)]
    bgs_o = [b if np.ndim(b) == 0 else embed(b, p)
             for b, p in zip(bgs, placements)]

    if init is None:
        init = _init_estimate(views, weights, bgs, scales, spacings, origins,
                              grid)

    res = rl_deconvolve_multiview(views_o, kernels_o, weights=weights_o,
                                  bg=bgs_o, view_scales=scales, init=init,
                                  return_info=return_info, **rl_kwargs)
    if not return_info:
        return res, grid

    u, info = res
    density = np.array([a / np.prod(p.steps)
                        for a, p in zip(info.view_scales, placements)])
    coverage = info.sensitivity / density.reshape((-1,) + (1,) * ndim)
    return u, grid, GridRLInfo(rl=info, coverage=coverage,
                               placements=placements, kernels=kernels_o)
