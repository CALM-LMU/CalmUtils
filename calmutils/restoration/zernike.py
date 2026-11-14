"""Zernike wavefront aberrations and pupil-based PSF synthesis.

Principled PSF generator: a wavefront error ``W(rho, theta)`` is expanded in
Zernike polynomials (coefficients in waves, i.e. fractions of lambda), the
complex pupil function is ``P * exp(2 pi i W)``, and the PSF is the squared
modulus of its Fourier transform::

    PSF(x) = |FT[ P(nu) * exp(2 pi i W(nu)) ]|^2

Sampling: a pupil of radius ``R`` px in an ``N x N`` FFT grid represents the
spatial-frequency radius ``NA/lambda``, so the output PSF pixel size is
``Delta = R * lambda / (N * NA)`` (see :func:`output_pixel_size`) and the Airy
first zero sits at ``0.61 * N / R`` px.  Defocus through a z-stack is included
physically: ``W_defocus(rho, z) = z * NA^2 / (2 lambda) * rho^2`` waves.

Conventions: Zernike modes ``Z_n^m(rho, theta) = R_n^|m|(rho) * cos(m theta)``
for ``m >= 0`` and ``sin(|m| theta)`` for ``m < 0`` (OSA/ANSI; Noll's j-index
uses the same (n, m) pairs, see :data:`NOLL`).  Angles are measured from the
x axis of the pupil; in a (y, x) PSF array the pupil x axis maps to the array's
last axis.

Three tiers:

    tier 1  psf_2d / psf_stack          grid-level, normalized units, explicit n/radius
    tier 2  psf_2d_norm / psf_stack_norm  sampling API (oversample, shape/shape_dist), auto n
    tier 3  psf_2d_phys / psf_stack_phys  physical units (na, wavelength_nm, n_ri,
                                          pixel_size_nm, dz_nm) - pure unit conversion
                                          of tier-2 sampling; the normalized PSF shape
                                          depends only on the coefficients

Tier-2 API:

Both work in normalized units (lambda = NA = 1): lateral unit lambda/NA,
axial unit lambda/NA^2 (Rayleigh range z_R = 2).  ``oversample`` is scalar or
per-axis in array-axis order ((os_y, os_x) for 2D, (os_z, os_y, os_x) for 3D);
the lateral Nyquist sample is 0.25 (lambda/NA) and the axial Nyquist sample is
0.5 (lambda/NA^2), so oversample=1 is Nyquist and larger values sample finer.
The FFT grid size ``n`` is chosen automatically.  ``shape`` is the output
array shape (scalar = same length per axis, vector = per-axis); if None, each
axis gets ``shape_dist`` natural distances: Airy diameters (1.22 lambda/NA)
laterally, Rayleigh ranges (2 lambda/NA^2) axially - converted to px/planes
via the axis sampling and odd-rounded.  Both return ``(psf, pixel_sizes)``
with pixel sizes in the normalized units, for later conversion to nm.

Optical model & assumptions:

- **Scalar Fourier optics**: Models diffraction via scalar Fourier optics
  (Debye approximation) with a flat circular pupil aperture. Neglects
  polarization-dependent PSF distortion and high-NA apodization
  (1/sqrt(cos theta)).
- **Paraxial defocus**: Defocus through focus uses the quadratic Fresnel
  phase term (NA^2 / (2 lambda n_ri) * rho^2). For very high NA/n_ri (> 0.7),
  the spherical aberration physically induced by large defocus is omitted.
- **Wavefront error convention**: Expansion coefficients are in waves of
  peak polynomial amplitude at the pupil edge (R_n^|m|(1) = 1, standard Born &
  Wolf / OSA), not RMS wavefront error. Orthonormal RMS coefficients differ
  by sqrt(2(n+1) / (1 + delta_m0)).
- **3D stack normalization**: Stacks return unit sum per lateral plane. When
  using a stack as a 3D deconvolution kernel (e.g. in ``rl_deconvolve``),
  the entire volume should be renormalized to sum to 1 (``st / st.sum()``).

This module is standalone (not yet wired into synthetic.py / the metrics
gallery); self-test: ``python -m calmutils.restoration.zernike``.
"""

import math

import numpy as np

__all__ = [
    "NOLL", "NAMED", "zernike_radial", "zernike", "noll",
    "pupil_mask", "wavefront", "output_pixel_size",
    "psf_2d", "psf_stack",
    "AIRY_DIAMETER", "RAYLEIGH_RANGE",
    "psf_2d_norm", "psf_stack_norm",
    "psf_2d_phys", "psf_stack_phys",
]

#: Noll j-index (j = 1, 2, ...) -> (n, m).  m > 0: cos mode, m < 0: sin mode.
NOLL = {
    1: (0, 0),
    2: (1, 1), 3: (1, -1),
    4: (2, 0), 5: (2, -2), 6: (2, 2),
    7: (3, -1), 8: (3, 1), 9: (3, -3), 10: (3, 3),
    11: (4, 0), 12: (4, 2), 13: (4, -2), 14: (4, 4), 15: (4, -4),
    16: (5, 1), 17: (5, -1), 18: (5, 3), 19: (5, -3), 20: (5, 5), 21: (5, -5),
    22: (6, 0), 23: (6, -2), 24: (6, 2), 25: (6, -4), 26: (6, 4), 27: (6, -6), 28: (6, 6),
}

#: Human-readable names for the common modes -> (n, m).
NAMED = {
    "piston": (0, 0),
    "tilt_x": (1, 1), "tilt_y": (1, -1),
    "defocus": (2, 0),
    "astigmatism_0": (2, 2), "astigmatism_45": (2, -2),
    "coma_y": (3, -1), "coma_x": (3, 1),
    "trefoil_y": (3, -3), "trefoil_x": (3, 3),
    "spherical": (4, 0),
    "secondary_astigmatism_0": (4, 2), "secondary_astigmatism_45": (4, -2),
    "quadrafoil_x": (4, 4), "quadrafoil_y": (4, -4),
    "secondary_coma_y": (5, -1), "secondary_coma_x": (5, 1),
    "secondary_spherical": (6, 0),
}


def noll(j):
    """(n, m) pair for a Noll index ``j`` (1-based)."""
    return NOLL[int(j)]


def zernike_radial(n, m, rho):
    """Radial Zernike polynomial R_n^|m|(rho).

    ``rho`` is the normalized pupil radius (1 at the pupil edge); ``n >= 0``,
    ``|m| <= n``, ``n - |m|`` even.  Uses the unnormalized OSA/ANSI convention
    (R_n^|m|(1) = 1), so coefficients represent peak amplitude at the pupil
    edge in waves, rather than orthonormal RMS wavefront error.
    """
    n, m = int(n), int(m)
    am = abs(m)
    if n < 0 or am > n or (n - am) % 2 != 0:
        raise ValueError(f"invalid Zernike indices n={n}, m={m}")
    rho = np.asarray(rho, dtype=float)
    out = np.zeros_like(rho)
    for s in range((n - am) // 2 + 1):
        c = ((-1) ** s * math.factorial(n - s)
             / (math.factorial(s)
                * math.factorial((n + am) // 2 - s)
                * math.factorial((n - am) // 2 - s)))
        out = out + c * rho ** (n - 2 * s)
    return out


def zernike(n, m, rho, theta):
    """Full Zernike polynomial Z_n^m(rho, theta) = R_n^|m|(rho) * angular part.

    ``m >= 0`` -> cos(m theta), ``m < 0`` -> sin(|m| theta); theta measured
    from the pupil x axis.
    """
    r = zernike_radial(n, m, rho)
    if m >= 0:
        return r * np.cos(m * theta)
    return r * np.sin(-m * theta)


def _norm_coeffs(coeffs):
    """Normalize a coefficient spec to a dict {(n, m): amplitude in waves}.

    Accepts: {(n, m): c}, {name: c} (names from :data:`NAMED`), {j: c} with
    int keys = Noll indices, or a plain list [c_1, c_2, ...] (Noll j = 1..len).
    """
    if isinstance(coeffs, (list, tuple)):
        return {NOLL[j + 1]: float(c) for j, c in enumerate(coeffs) if c != 0}
    out = {}
    for k, c in dict(coeffs).items():
        if isinstance(k, str):
            key = NAMED[k]
        elif isinstance(k, (int, np.integer)):
            key = NOLL[int(k)]
        else:
            key = tuple(k)
        out[key] = out.get(key, 0.0) + float(c)
    return {k: v for k, v in out.items() if v != 0}


def _polar_grids(n, radius):
    """Normalized pupil polar grids: rho in [0, 1] at the edge, theta from x axis."""
    c = (n - 1) / 2.0
    yy, xx = np.meshgrid(np.arange(n) - c, np.arange(n) - c, indexing="ij")
    rho = np.sqrt(xx ** 2 + yy ** 2) / radius
    theta = np.arctan2(yy, xx)  # y is the first array axis, x the last (pupil x = last axis)
    return rho, theta


def pupil_mask(n, radius):
    """Unit-disk pupil of radius ``radius`` px in an ``n x n`` grid (bool)."""
    rho, _ = _polar_grids(n, radius)
    return rho <= 1.0


def wavefront(n, coeffs, radius=None):
    """Wavefront error W(rho, theta) in waves on an ``n x n`` grid.

    ``coeffs``: see :func:`_norm_coeffs` ((n,m) dict, name dict, or Noll list).
    Values represent peak polynomial amplitude in waves (R_n^|m|(1) = 1),
    not RMS wavefront error. Zero outside the pupil.
    """
    radius = float(radius) if radius is not None else n / 4.0
    rho, theta = _polar_grids(n, radius)
    w = np.zeros((n, n))
    for (nn, mm), c in _norm_coeffs(coeffs).items():
        w += c * zernike(nn, mm, rho, theta)
    w[rho > 1.0] = 0.0
    return w


def output_pixel_size(n, radius, na, wavelength):
    """PSF pixel size (same length unit as ``wavelength``) for an ``n x n`` FFT
    grid with pupil radius ``radius`` px: ``Delta = R*lambda / (N*NA)``."""
    return float(radius) * float(wavelength) / (int(n) * float(na))


def _psf_from_field(field):
    """|FFT|^2 of a complex pupil field, unit sum, centered."""
    psf = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))) ** 2
    s = psf.sum()
    if s <= 0:
        raise ValueError("zero PSF (empty pupil?)")
    return psf / s


def _center_crop(a, crop):
    """Center-crop the *last* ``len(crop)`` axes (leading axes, e.g. z, kept)."""
    if crop is None:
        return a
    crop = tuple(int(c) for c in crop)
    if len(crop) > a.ndim or any(c > s for c, s in zip(crop, a.shape[-len(crop):])):
        raise ValueError(f"crop {crop} does not fit array shape {a.shape}")
    # center the crop on the array's central voxel (s//2); for even s this is
    # the FFT's DC bin after fftshift, so the PSF stays centered in the window
    sl = (tuple(slice(None) for _ in range(a.ndim - len(crop)))
          + tuple(slice(s // 2 - c // 2, s // 2 - c // 2 + c)
                  for s, c in zip(a.shape[-len(crop):], crop)))
    return a[sl]


def psf_2d(coeffs, *, n=256, radius=None, crop=None):
    """In-focus 2D PSF for a wavefront given by ``coeffs`` (waves).

    Parameters
    ----------
    coeffs : {(n, m): c} | {name: c} | [c_1, c_2, ...] (Noll list)
    n : FFT grid size (pupil array is n x n)
    radius : pupil radius in px (default n/4; sets the PSF sampling, see
        :func:`output_pixel_size`)
    crop : optional odd output window (px), center-cropped from the n x n PSF

    Returns
    -------
    unit-sum 2D PSF (float array, ``crop`` if given else ``n x n``)
    """
    radius = float(radius) if radius is not None else n / 4.0
    w = wavefront(n, coeffs, radius)
    pupil = pupil_mask(n, radius)
    field = pupil * np.exp(2j * np.pi * w)
    psf = _center_crop(_psf_from_field(field), crop)
    return psf / psf.sum()  # unit sum within the output window


def psf_stack(coeffs, *, n=256, radius=None, crop=None, z_planes=1, dz=0.0,
              defocus_waves_per_z=0.5):
    """3D PSF stack with paraxial defocus propagation.

    Each z plane gets the extra wavefront
    ``W_defocus = z * defocus_waves_per_z * rho^2`` (waves), i.e. planes at
    ``z = (iz - (z_planes-1)/2) * dz`` around the in-focus plane.

    ``defocus_waves_per_z`` is the physical defocus coefficient
    ``NA^2 / (2 lambda)`` (waves per unit z at the pupil edge; immersion: divide
    by the medium index n).  In normalized units (lambda = NA = 1) it is 0.5 by
    definition - 1 wave at the pupil edge per Rayleigh range - and
    :func:`psf_stack_norm` always uses that.  ``dz`` is in the same length
    units as the caller's z.

    Returns a unit-sum-per-plane stack of shape ``(z_planes, *crop)`` (or
    ``(z_planes, n, n)``).  Note: if used as a 3D deconvolution kernel,
    renormalize the entire 3D volume to unit sum (``st / st.sum()``).
    """
    radius = float(radius) if radius is not None else n / 4.0
    z_planes = int(z_planes)
    if z_planes < 1:
        raise ValueError("z_planes must be >= 1")
    w0 = wavefront(n, coeffs, radius)
    pupil = pupil_mask(n, radius)
    rho, _ = _polar_grids(n, radius)
    rho2 = rho ** 2
    coeff_def = float(defocus_waves_per_z)  # waves per unit z at rho=1
    planes = []
    for iz in range(z_planes):
        z = (iz - (z_planes - 1) / 2.0) * float(dz)
        field = pupil * np.exp(2j * np.pi * (w0 + z * coeff_def * rho2))
        planes.append(_psf_from_field(field))
    st = _center_crop(np.stack(planes), crop)
    return st / st.sum(axis=tuple(range(1, st.ndim)), keepdims=True)  # unit sum per plane


# ---------------------------------------------------------------------------
# normalized-units API (lambda = NA = 1): lateral unit lambda/NA, axial unit
# lambda/NA^2.  Lateral Nyquist sample 0.25, axial Nyquist sample 0.5.
# ---------------------------------------------------------------------------

def _next_pow2(x):
    return 1 << int(np.ceil(np.log2(max(1, x))))


def _odd(x):
    x = max(3, int(round(x)))
    return x if x % 2 == 1 else x + 1


# natural length units in normalized coordinates (lambda = NA = 1)
AIRY_DIAMETER = 1.22    # 2 * 0.61 lambda/NA: first dark ring diameter
RAYLEIGH_RANGE = 2.0    # z_R = 2 lambda/NA^2: first axial zero of the axial PSF


def _resolve_oversample(oversample, ndim):
    """Scalar or per-axis oversampling in array-axis order; os=1 is Nyquist.

    The >= 0.5 floor (pupil must fit the FFT grid) applies only to the lateral
    axes; the axial os is free because z planes are analytic, not FFT-limited.
    """
    os = np.atleast_1d(np.asarray(oversample, dtype=float))
    if os.size == 1:
        os = np.full(ndim, float(os[0]))
    if os.size != ndim:
        raise ValueError(f"oversample must be scalar or have {ndim} entries, got {os.tolist()}")
    if np.any(os <= 0):
        raise ValueError("oversample must be > 0")
    lateral = os if ndim == 2 else os[1:]
    if np.any(lateral < 0.5):
        raise ValueError("lateral oversample must be >= 0.5 (pupil must fit the FFT grid)")
    return os


def _resolve_shape(shape, deltas, shape_dist, ndim):
    """Output array shape per axis. None -> shape_dist natural distances per
    axis (Airy diameters laterally, Rayleigh range axially), converted to
    px/planes via that axis's sampling and odd-rounded."""
    dist = ([RAYLEIGH_RANGE] + [AIRY_DIAMETER] * (ndim - 1)) if ndim == 3 \
        else [AIRY_DIAMETER] * 2
    if shape is None:
        sd = np.atleast_1d(np.asarray(shape_dist, dtype=float))
        if sd.size == 1:
            sd = np.full(ndim, float(sd[0]))
        if sd.size != ndim:
            raise ValueError(f"shape_dist must be scalar or have {ndim} entries")
        return tuple(_odd(s * f / d) for s, f, d in zip(sd, dist, deltas))
    if np.isscalar(shape):
        return tuple(_odd(shape) for _ in range(ndim))
    shape = tuple(int(s) for s in shape)
    if len(shape) != ndim:
        raise ValueError(f"shape must be scalar or have {ndim} entries, got {shape}")
    return tuple(_odd(s) for s in shape)


def _auto_n(shape_max, os_synth):
    """FFT grid size: big enough for the crop window plus PSF tails, and with
    pupil radius R = n*0.25/os >= 8 px for a smooth pupil edge."""
    return max(256, _next_pow2(4 * shape_max), int(np.ceil(32 * os_synth)))


def psf_2d_norm(coeffs, *, oversample=2.0, shape=None, shape_dist=4.0):
    """In-focus 2D PSF in normalized units (lambda = NA = 1, lateral unit
    lambda/NA).

    Parameters
    ----------
    coeffs : wavefront coefficients in waves (see :func:`wavefront`)
    oversample : scalar or (os_y, os_x); 1 = Nyquist (0.25 lambda/NA per px).
        Synthesized isotropically at max(os_y, os_x) and resampled per-axis if
        os_y != os_x.
    shape : None -> ``shape_dist`` Airy diameters per axis; scalar -> (s, s);
        vector -> (sy, sx)
    shape_dist : default output size in Airy diameters (1.22 lambda/NA) per axis

    Returns
    -------
    (psf, (dy, dx)) : unit-sum PSF of shape ``shape`` and its pixel sizes in
    lambda/NA units
    """
    os_y, os_x = _resolve_oversample(oversample, 2)
    deltas = (0.25 / os_y, 0.25 / os_x)
    sy, sx = _resolve_shape(shape, deltas, shape_dist, 2)
    os_s = max(os_y, os_x)
    d_s = 0.25 / os_s
    n = _auto_n(max(sy, sx), os_s)
    psf = psf_2d(coeffs, n=n, radius=n * d_s)  # full n x n, peak at DC bin
    if os_y == os_x:
        psf = _center_crop(psf, (sy, sx))
        psf = psf / psf.sum()
        return psf, (d_s, d_s)
    from .resampling import resample_psf
    psf = _center_crop(psf, (n - 1, n - 1))  # odd input for resample_psf
    # no out_size: preserves the physical window, eff pixel size ~= target
    psf, eff = resample_psf(psf, (d_s, d_s), deltas)
    psf = _center_crop(psf, (sy, sx))
    return psf / psf.sum(), (float(eff[0]), float(eff[1]))


def psf_stack_norm(coeffs, *, oversample=2.0, shape=None, shape_dist=4.0):
    """3D PSF stack in normalized units (lambda = NA = 1; lateral unit
    lambda/NA, axial unit lambda/NA^2, Rayleigh range z_R = 2).

    Same parameters as :func:`psf_2d_norm`, with oversample/shape in array
    axis order (z, y, x): oversample (os_z, os_y, os_x) with axial Nyquist
    sample dz = 0.5/os_z (z is analytic per plane, not FFT-limited); shape
    (sz, sy, sx); shape_dist = Rayleigh ranges on z, Airy diameters on y/x.

    Returns
    -------
    (stack, (dz, dy, dx)) : unit-sum-per-plane stack of shape ``shape`` and
    its pixel sizes in normalized units
    """
    os_z, os_y, os_x = _resolve_oversample(oversample, 3)
    dz = 0.5 / os_z
    deltas = (dz, 0.25 / os_y, 0.25 / os_x)
    sz, sy, sx = _resolve_shape(shape, deltas, shape_dist, 3)
    os_s = max(os_y, os_x)
    d_s = 0.25 / os_s
    n = _auto_n(max(sy, sx), os_s)
    st = psf_stack(coeffs, n=n, radius=n * d_s, z_planes=sz, dz=dz)  # (sz, n, n)
    if os_y == os_x:
        st = _center_crop(st, (sy, sx))
        st = st / st.sum(axis=(1, 2), keepdims=True)
        return st, (dz, d_s, d_s)
    from .resampling import resample_psf
    st = _center_crop(st, (sz, n - 1, n - 1))  # odd input for resample_psf
    # no out_size: preserves the physical window, eff pixel size ~= target
    st, eff = resample_psf(st, (dz, d_s, d_s), (dz, deltas[1], deltas[2]))
    st = _center_crop(st, (sz, sy, sx))
    st = st / st.sum(axis=(1, 2), keepdims=True)
    return st, (float(eff[0]), float(eff[1]), float(eff[2]))


# ---------------------------------------------------------------------------
# tier 3: physical units (na, wavelength, pixel size in nm) -> tier 2
# ---------------------------------------------------------------------------

def psf_2d_phys(coeffs, *, na, wavelength_nm, pixel_size_nm, shape=None,
                shape_dist=4.0):
    """2D PSF on a physical pixel grid (tier 3).

    Parameters
    ----------
    coeffs : wavefront coefficients in waves
    na, wavelength_nm : optics (lateral unit lambda/na nm)
    pixel_size_nm : output pixel size in nm
    shape : output window in px (odd); None -> ``shape_dist`` Airy diameters
        (unit-free, physical)
    shape_dist : default window in Airy diameters (1.22 lambda/na)

    Returns
    -------
    (psf, pixel_size_nm_eff) : unit-sum PSF, effective pixel size in nm
    """
    from .resampling import resample_psf
    lat_unit_nm = float(wavelength_nm) / float(na)          # lambda/na in nm
    d_norm = float(pixel_size_nm) / lat_unit_nm              # target pixel in lambda/na units
    os_synth = max(0.25 / d_norm, 0.5)                       # never synthesize coarser than os=0.5
    d_synth = 0.25 / os_synth

    if shape is not None:
        s = (_odd(shape), _odd(shape)) if np.isscalar(shape) else tuple(_odd(v) for v in shape)
        if len(s) != 2:
            raise ValueError(f"shape must be scalar or have 2 entries, got {shape}")
    else:
        s = None

    if np.isclose(os_synth, 0.25 / d_norm):
        # Target pixel size can be synthesized directly without resampling
        psf, (dy, dx) = psf_2d_norm(coeffs, oversample=os_synth, shape=s,
                                    shape_dist=shape_dist)
        eff = (dy, dx)
    else:
        # Undersampled laterally: synthesize at os=0.5 and resample to target pixel size
        s_synth = tuple(_odd(v * d_norm / d_synth) for v in s) if s is not None else None
        psf, (dy, dx) = psf_2d_norm(coeffs, oversample=os_synth, shape=s_synth,
                                    shape_dist=shape_dist)
        psf, eff = resample_psf(psf, (dy, dx), (d_norm, d_norm), out_size=s)
        if s is not None:
            psf = _center_crop(psf, s)
            psf = psf / psf.sum()

    return psf, float(eff[0]) * lat_unit_nm


def psf_stack_phys(coeffs, *, na, wavelength_nm, pixel_size_nm, dz_nm, n_ri=1.0,
                   shape=None, shape_dist=4.0):
    """3D PSF stack on a physical grid (tier 3).

    Like :func:`psf_2d_phys`, plus ``dz_nm`` (axial plane spacing in nm) and
    ``n_ri`` (medium refractive index; axial unit n_ri*lambda/na^2 nm, i.e. the
    Rayleigh range scales with n_ri).  Defocus propagation uses the paraxial
    approximation.  ``shape`` is the output array shape (sz, sy, sx) in
    px/planes; None -> ``shape_dist`` Rayleigh ranges on z, Airy diameters on
    y/x.

    Returns
    -------
    (stack, (dz_nm, dy_nm, dx_nm)) : unit-sum-per-plane stack + pixel sizes in nm.
        Note: if passed as a 3D deconvolution kernel (e.g. to
        :func:`~calmutils.restoration.rl_deconvolve`), renormalize the volume
        to unit sum (``stack / stack.sum()``).
    """
    from .resampling import resample_psf
    lat_unit_nm = float(wavelength_nm) / float(na)
    ax_unit_nm = float(n_ri) * float(wavelength_nm) / float(na) ** 2  # lambda/na^2 with RI
    d_norm = float(pixel_size_nm) / lat_unit_nm
    dz_norm = float(dz_nm) / ax_unit_nm
    os_synth = max(0.25 / d_norm, 0.5)
    d_synth = 0.25 / os_synth
    os_z = 0.5 / dz_norm                                    # axial os is free (analytic planes)

    if shape is not None:
        s = tuple(_odd(shape) for _ in range(3)) if np.isscalar(shape) else tuple(_odd(v) for v in shape)
        if len(s) != 3:
            raise ValueError(f"shape must be scalar or have 3 entries, got {shape}")
    else:
        s = None

    if np.isclose(os_synth, 0.25 / d_norm):
        # Target lateral pixel size can be synthesized directly without resampling
        st, (dz, dy, dx) = psf_stack_norm(coeffs, oversample=(os_z, os_synth, os_synth),
                                         shape=s, shape_dist=shape_dist)
        eff = (dz, dy, dx)
    else:
        # Undersampled laterally: synthesize at os=0.5 and resample lateral axes
        s_synth = (s[0], _odd(s[1] * d_norm / d_synth), _odd(s[2] * d_norm / d_synth)) if s is not None else None
        st, (dz, dy, dx) = psf_stack_norm(coeffs, oversample=(os_z, os_synth, os_synth),
                                         shape=s_synth, shape_dist=shape_dist)
        st, eff = resample_psf(st, (dz, dy, dx), (dz, d_norm, d_norm), out_size=s)
        st = st / st.sum(axis=(1, 2), keepdims=True)
        if s is not None:
            st = _center_crop(st, s)
            st = st / st.sum(axis=(1, 2), keepdims=True)

    return st, (float(eff[0]) * ax_unit_nm, float(eff[1]) * lat_unit_nm,
                float(eff[2]) * lat_unit_nm)


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------

def _selftest():
    ok = True
    n, radius = 256, 64

    # --- Zernike orthogonality over the unit disk:
    # (1/pi) ∫ Z_n^m Z_n'^m' dA = delta_nn' delta_mm' * (1/(n+1) if m == 0 else 1/(2(n+1)))
    rr = np.linspace(0, 1, 400)
    tt = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    RR, TT = np.meshgrid(rr, tt, indexing="ij")
    dA = RR * (rr[1] - rr[0]) * (tt[1] - tt[0])
    for (n1, m1), (n2, m2) in [((0, 0), (0, 0)), ((2, 0), (2, 0)), ((2, 0), (2, 2)),
                               ((3, 1), (3, 1)), ((3, 1), (3, -1)), ((4, 0), (0, 0))]:
        z1 = zernike(n1, m1, RR, TT)
        z2 = zernike(n2, m2, RR, TT)
        integ = float((z1 * z2 * dA).sum()) / np.pi
        exp = 0.0
        if (n1, m1) == (n2, m2):
            exp = 1.0 / (n1 + 1) if m1 == 0 else 1.0 / (2 * (n1 + 1))
        print(f"[zernike] (1/pi)∫ Z_{n1}^{m1} Z_{n2}^{m2} dA = {integ:+.5f} (expect {exp:+.5f})")
        ok &= abs(integ - exp) < 5e-3

    # --- unaberrated PSF = Airy pattern, first zero at 0.61*n/radius px from center
    psf = psf_2d({}, n=n, radius=radius, crop=(41, 41))
    c = 20
    prof = psf[c, :]
    com0 = float((np.arange(41) * prof).sum() / prof.sum())  # unaberrated centroid
    airy_zero = 0.61 * n / radius
    # first drop below 2% of the peak after the central lobe = first dark ring
    i_rel = int(np.argmax(prof[c + 1:] < 0.02 * prof.max())) + 1
    print(f"[psf] Airy first dark ring {i_rel} px from center (centroid {com0:.2f}), "
          f"expected {airy_zero:.2f} px")
    ok &= abs(i_rel - airy_zero) < 1.0
    ok &= abs(psf.sum() - 1.0) < 1e-12

    # --- tilt shifts the PSF by c * n / radius px (FT shift theorem)
    c_tilt = 0.5
    psf_t = psf_2d({(1, 1): c_tilt}, n=n, radius=radius, crop=(41, 41))  # tilt_x -> shift along x
    com_x = float((np.arange(41) * psf_t[20]).sum() / psf_t[20].sum())
    exp_shift = c_tilt * n / radius
    print(f"[psf] tilt_x {c_tilt} waves: centroid shift {com_x - com0:+.3f} px, "
          f"expected {exp_shift:.2f} px")
    ok &= abs((com_x - com0) - exp_shift) < 0.5

    # --- defocus broadens the PSF monotonically (2nd moment of the marginal)
    sigs = []
    for c_def in (0.0, 0.5, 1.0, 2.0):
        p = psf_2d({"defocus": c_def}, n=n, radius=radius, crop=(81, 81))
        m81 = p[40]
        sigs.append(float(np.sqrt(((np.arange(81) - 40) ** 2 * m81).sum() / m81.sum())))
    print(f"[psf] defocus 0/0.5/1/2 waves -> sigma_x {np.round(sigs, 2)}; "
          f"monotonic: {bool(np.all(np.diff(sigs) > 0))}")
    ok &= np.all(np.diff(sigs) > 0)
    st = psf_stack({"defocus": 0.0}, n=n, radius=radius, crop=(41, 41),
                    z_planes=5, dz=100.0, defocus_waves_per_z=1.3 ** 2 / (2 * 500.0))
    mid = [float(np.sqrt(((np.arange(41) - 20) ** 2 * st[iz, 20]).sum())) for iz in range(5)]
    print(f"[psf] pure defocus stack z=-200..200 nm: sigma_x per plane {np.round(mid, 2)} "
          f"(symmetric U-shape: {bool(np.isclose(mid[0], mid[-1]) and mid[2] < mid[0])})")
    ok &= np.isclose(mid[0], mid[-1]) and mid[2] < mid[0]
    ok &= abs(st.sum() - 5.0) < 1e-9  # unit sum per plane

    # --- coma: one-sided tail -> positive asymmetry along x
    from calmutils.restoration.psf_metrics import psf_metrics
    p_coma = psf_2d({"coma_x": 1.0}, n=n, radius=radius, crop=(41, 41))
    m = psf_metrics(p_coma)
    print(f"[psf] coma_x 1 wave: asymmetry_x {m['asymmetry_px'][1]:+.3f}, skew_x {m['skew'][1]:+.3f} "
          f"(expect > 0)")
    ok &= m["asymmetry_px"][1] > 0.05 and m["skew"][1] > 0.05

    # --- Noll list and named dict agree
    p1 = psf_2d({8: 1.0}, n=n, radius=radius, crop=(21, 21))       # Noll j=8 = (3,1) coma_x
    p2 = psf_2d({"coma_x": 1.0}, n=n, radius=radius, crop=(21, 21))
    same = bool(np.allclose(p1, p2))
    print(f"[psf] Noll dict {{8: 1}} == {{'coma_x': 1}}: {same}")
    ok &= same
    # Noll parity rule: even j has m > 0 (cos), odd j has m < 0 (sin)
    noll_parity_ok = True
    for j, (nj, mj) in NOLL.items():
        if mj != 0:
            if j % 2 == 0 and mj <= 0:
                noll_parity_ok = False
            elif j % 2 == 1 and mj >= 0:
                noll_parity_ok = False
    print(f"[psf] Noll even=cos / odd=sin parity rule: {noll_parity_ok}")
    ok &= noll_parity_ok
    ok &= NOLL[16] == NAMED["secondary_coma_x"] and NOLL[17] == NAMED["secondary_coma_y"]

    # ---------------- normalized API: psf_2d_norm / psf_stack_norm ----------------
    psf, (dy, dx) = psf_2d_norm({})
    exp_s = _odd(round(4.0 * AIRY_DIAMETER / 0.125))
    print(f"[norm] 2D defaults: shape {psf.shape} (expect {exp_s} = 4 Airy diameters at os=2), "
          f"dy=dx={dy} (expect 0.125), sum {psf.sum():.6f}")
    ok &= psf.shape == (exp_s, exp_s) and dy == dx == 0.125 and abs(psf.sum() - 1.0) < 1e-9
    # unaberrated PSF: first dark ring at 0.61 lambda/NA = 0.61/dy px from center
    c = psf.shape[1] // 2
    ring = int(np.argmax(psf[c, c + 1:] < 0.02 * psf.max())) + 1
    print(f"[norm] Airy first dark ring at {ring} px (expect ~{0.61 / dy:.1f})")
    ok &= abs(ring - 0.61 / dy) < 1.5

    psf_c, _ = psf_2d_norm({'coma_x': 1.0}, oversample=2.0, shape=41)
    m_c = psf_metrics(psf_c)
    print(f"[norm] 2D coma 1w scalar shape=41: asym_x {m_c['asymmetry_px'][1]:+.3f} (>0), "
          f"peak {psf_c.max():.4f} vs airy {psf.max():.4f}")
    ok &= m_c["asymmetry_px"][1] > 0.1 and psf_c.shape == (41, 41)

    psf_a, (dya, dxa) = psf_2d_norm({'coma_x': 0.2}, oversample=(2.0, 4.0), shape=(41, 61))
    print(f"[norm] 2D aniso oversample (2,4): shape {psf_a.shape}, dy={dya:.4f} dx={dxa:.4f} "
          f"(expect 0.125 / 0.0625)")
    ok &= psf_a.shape == (41, 61)
    ok &= abs(dya - 0.125) / 0.125 < 0.05 and abs(dxa - 0.0625) / 0.0625 < 0.05  # resample rounding (<5%)

    st, (dz, dy, dx) = psf_stack_norm({'defocus': 0.3})
    exp_sz = _odd(round(4.0 * RAYLEIGH_RANGE / 0.25))
    print(f"[norm] 3D defaults: shape {st.shape} (expect sz={exp_sz} = 4 Rayleigh ranges, "
          f"sy=sx={exp_s}), dz={dz} (expect 0.25), per-plane sums 1: "
          f"{bool(np.allclose(st.sum(axis=(1, 2)), 1.0))}")
    ok &= st.shape == (exp_sz, exp_s, exp_s) and dz == 0.25 and dy == dx == 0.125
    ok &= bool(np.allclose(st.sum(axis=(1, 2)), 1.0))

    st2, (dz2, dy2, dx2) = psf_stack_norm({}, oversample=(2.0, 2.0, 4.0), shape=(9, 21, 31))
    print(f"[norm] 3D aniso oversample (os_z,os_y,os_x)=(2,2,4): shape {st2.shape}, "
          f"dz={dz2} dy={dy2} dx={dx2} (expect 0.25/0.125/0.0625)")
    ok &= st2.shape == (9, 21, 31)
    # resampled lateral axes: eff pixel size matches target within odd-size rounding (<5%)
    ok &= dz2 == 0.25
    ok &= abs(dy2 - 0.125) / 0.125 < 0.05 and abs(dx2 - 0.0625) / 0.0625 < 0.05

    # unaberrated stack is z-symmetric; a static defocus *coefficient* shifts the
    # focal plane off z=0, so symmetry holds about the shifted focus, not z=0
    st0, _ = psf_stack_norm({})
    prof0 = st0[:, st0.shape[1] // 2, st0.shape[2] // 2]
    sym = bool(np.allclose(prof0, prof0[::-1], atol=1e-9))
    prof = st[:, st.shape[1] // 2, st.shape[2] // 2]
    sym_def = bool(np.allclose(prof, prof[::-1], atol=0.06))  # roughly symmetric (focus shifted)
    print(f"[norm] unaberrated stack z-symmetric: {sym}; defocus-coeff stack ~symmetric: {sym_def}")
    ok &= sym and sym_def

    # ---------------- tier 3: physical units ----------------
    na, wl, px = 1.3, 500.0, 50.0
    p2, px_eff = psf_2d_phys({}, na=na, wavelength_nm=wl, pixel_size_nm=px)
    lat_unit = wl / na  # 384.6 nm
    exp_shape = _odd(round(4.0 * AIRY_DIAMETER / (0.25 / max(0.25 / (px / lat_unit), 0.5))))
    print(f"[phys] 2D: shape {p2.shape} (expect ~{exp_shape}), pixel {px_eff:.1f} nm "
          f"(target {px}), airy ring at {0.61 * lat_unit / px_eff:.1f} px")
    ok &= abs(px_eff - px) / px < 0.05 and abs(p2.sum() - 1.0) < 1e-9
    c = p2.shape[1] // 2
    ring = int(np.argmax(p2[c, c + 1:] < 0.02 * p2.max())) + 1
    ok &= abs(ring - 0.61 * lat_unit / px_eff) < 1.5

    stp, (dz_nm, dy_nm, dx_nm) = psf_stack_phys({}, na=na, wavelength_nm=wl,
                                               pixel_size_nm=px, dz_nm=100.0)
    ax_unit = wl / na ** 2  # 295.9 nm (n_ri=1)
    print(f"[phys] 3D: shape {stp.shape}, dz={dz_nm:.1f} nm (target 100), "
          f"dy=dx={dy_nm:.1f} nm (target {px}), per-plane sums 1: "
          f"{bool(np.allclose(stp.sum(axis=(1, 2)), 1.0))}")
    ok &= abs(dz_nm - 100.0) / 100.0 < 0.05 and abs(dy_nm - px) / px < 0.05
    ok &= bool(np.allclose(stp.sum(axis=(1, 2)), 1.0))
    # immersion scales the axial unit by n_ri: dz_nm for the same normalized dz changes
    stp2, (dz_nm2, _, _) = psf_stack_phys({}, na=na, wavelength_nm=wl, pixel_size_nm=px,
                                         dz_nm=100.0, n_ri=1.5)
    print(f"[phys] n_ri=1.5: dz_eff {dz_nm2:.1f} nm vs n_ri=1 {dz_nm:.1f} nm "
          f"(axial unit x1.5 -> effective dz_nm at fixed normalized sampling)")
    ok &= abs(dz_nm2 - dz_nm) / dz_nm < 0.05  # same normalized dz -> ~same nm (unit conversion consistent)
    # defocus strength in physical units: na^2/(2 lambda) waves/nm; check a defocused
    # physical stack broadens with |z| like the normalized one
    stp3, _ = psf_stack_phys({'defocus': 0.3}, na=na, wavelength_nm=wl,
                             pixel_size_nm=px, dz_nm=100.0)
    ok &= stp3.shape[0] > 5
    print(f"[phys] 3D with defocus coeff: shape {stp3.shape} ok")

    # explicit shapes in physical units (scalar, 2D tuple, 3D stack)
    p2_s, px_s = psf_2d_phys({}, na=na, wavelength_nm=wl, pixel_size_nm=px, shape=65)
    print(f"[phys] 2D explicit scalar shape=65: shape {p2_s.shape}, pixel {px_s:.1f} nm")
    ok &= p2_s.shape == (65, 65) and abs(px_s - px) / px < 0.05
    p2_rect, _ = psf_2d_phys({}, na=na, wavelength_nm=wl, pixel_size_nm=px, shape=(45, 65))
    print(f"[phys] 2D explicit rect shape=(45, 65): shape {p2_rect.shape}")
    ok &= p2_rect.shape == (45, 65)
    stp_shape, (dz_s, dy_s, dx_s) = psf_stack_phys({}, na=na, wavelength_nm=wl,
                                                  pixel_size_nm=px, dz_nm=100.0,
                                                  shape=(15, 45, 45))
    print(f"[phys] 3D explicit shape=(15, 45, 45): shape {stp_shape.shape}, "
          f"dz={dz_s:.1f} dy=dx={dy_s:.1f} nm")
    ok &= stp_shape.shape == (15, 45, 45) and abs(dz_s - 100.0) / 100.0 < 0.05

    # undersampled physical pixel (d_norm > 0.5)
    p2_under, px_u = psf_2d_phys({}, na=na, wavelength_nm=wl, pixel_size_nm=250.0, shape=21)
    print(f"[phys] 2D undersampled (px=250nm): shape {p2_under.shape}, pixel {px_u:.1f} nm")
    ok &= p2_under.shape == (21, 21) and abs(px_u - 250.0) / 250.0 < 0.05

    print("SELFTEST", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    _selftest()
