"""Interactive napari viewer for the Zernike PSF synthesis in
``calmutils.restoration.zernike``.

Run it with an interpreter that has napari + magicgui installed (the ``viz``
optional extra), e.g.:

    /Users/david/miniforge3/envs/env-py314/bin/python zernike_examples/zernike_psf_viewer.py

A docked magicgui widget (see
https://napari.org/stable/howtos/extending/magicgui.html) exposes sliders for
the coefficients (in waves) of a few prominent Zernike modes, plus the optical
parameters (NA, wavelength, pixel size, display window).  Two layers update
live as the sliders move:

- ``Wavefront``: the wavefront error W(rho, theta) in waves over the pupil
  (diverging colormap), on the FFT pupil grid.
- ``PSF``: the resulting in-focus 2D PSF from
  :func:`calmutils.restoration.zernike.psf_2d_phys` on a physical pixel grid
  (nm scale bar).  The raw (unit-sum) array is assigned to the layer; contrast
  limits and gamma are deliberately *not* touched, so napari's auto-contrast
  (re-applied on every ``layer.data`` assignment as long as the user hasn't
  overridden it) and the native per-layer gamma slider stay in control.
- ``PSF3D`` (only while the "3D PSF stack" checkbox is on): the 3D PSF from
  :func:`calmutils.restoration.zernike.psf_stack_phys` with axial spacing
  ``dz_nm``; the viewer switches to 3D rendering while the layer exists.

The widget also shows live PSF metrics (from
:func:`calmutils.restoration.psf_metrics.psf_metrics`) of the in-focus 2D PSF.
Each metric's text is colored with the same diverging colormap as the
``Wavefront`` layer (vispy ``RdBu``), encoding the signed deviation from the
unaberrated (Airy) PSF at the same optics - so you can see at a glance how a
coefficient change moves each metric relative to the diffraction-limited case.
"""

import napari
import numpy as np
from magicgui import magicgui
from magicgui.widgets import Label, PushButton
from vispy.color import get_colormap

# note: calmutils.restoration re-exports a `zernike` *function*, which shadows
# the submodule, so import the functions from the submodule path directly
from calmutils.restoration.zernike import psf_2d_phys, psf_stack_phys, wavefront
from calmutils.restoration.psf_metrics import psf_metrics

# Prominent Zernike modes exposed as coefficient sliders: name -> (min, max)
# in waves (peak polynomial amplitude at the pupil edge, OSA convention).
MODE_SLIDERS = {
    "defocus": (-3.0, 3.0),
    "astigmatism_0": (-2.0, 2.0),
    "astigmatism_45": (-2.0, 2.0),
    "coma_x": (-2.0, 2.0),
    "coma_y": (-2.0, 2.0),
    "trefoil_x": (-1.0, 1.0),
    "spherical": (-1.0, 1.0),
}

# pupil grid used only for the wavefront display layer
WAVEFRONT_N = 256
WAVEFRONT_RADIUS = 64

# diverging colormap for the metric labels - same one as the Wavefront layer
_DIVERGING = get_colormap("RdBu")

# metrics shown in the widget: key into _metric_values(), display name, how the
# deviation from the Airy baseline is measured ("rel" = v/b - 1, "diff" = v - b)
# and the deviation magnitude that maps to full colormap saturation
METRIC_SPECS = [
    {"key": "peak", "name": "peak (Strehl)", "mode": "rel", "scale": 0.5},
    {"key": "fwhm", "name": "FWHM mean", "mode": "rel", "scale": 0.2, "unit": " nm"},
    {"key": "sigma", "name": "sigma mean", "mode": "rel", "scale": 0.2, "unit": " nm"},
    {"key": "sidelobe", "name": "sidelobe ratio", "mode": "diff", "scale": 0.1},
    {"key": "asym_x", "name": "asymmetry x", "mode": "diff", "scale": 0.2},
    {"key": "asym_y", "name": "asymmetry y", "mode": "diff", "scale": 0.2},
    {"key": "skew_x", "name": "skew x", "mode": "diff", "scale": 0.5},
    {"key": "skew_y", "name": "skew y", "mode": "diff", "scale": 0.5},
    {"key": "astig", "name": "astigmatism", "mode": "diff", "scale": 0.1},
    {"key": "elong", "name": "elongation", "mode": "diff", "scale": 0.5},
]

# cache of the Airy-baseline metric values per optical setting
_baseline_cache = {}


def _metric_values(psf, px_nm, m):
    """Flat dict of display values from a psf_metrics() result."""
    astig = m["astigmatism"]
    return {
        "peak": float(psf.max()),
        "fwhm": float(np.mean(m["fwhm_phys"])),
        "sigma": float(np.mean(m["sigma_px"])) * px_nm,
        "sidelobe": float(m["sidelobe_ratio"]),
        "asym_x": float(m["asymmetry_px"][1]),
        "asym_y": float(m["asymmetry_px"][0]),
        "skew_x": float(m["skew"][1]),
        "skew_y": float(m["skew"][0]),
        "astig": float(astig) if astig is not None else float("nan"),
        "elong": float(m["elongation"]),
    }


def _deviation_color(dev, scale):
    """Signed deviation (in units of ``scale``) -> hex color on the diverging
    RdBu map, clipped to full saturation at |dev| = scale. dev < 0 -> red side,
    dev > 0 -> blue side, 0 -> white (same orientation as the Wavefront layer).
    """
    if not np.isfinite(dev):
        dev = 0.0
    t = float(np.clip(dev / scale, -1.0, 1.0))
    rgba = _DIVERGING.map(np.asarray([(t + 1.0) / 2.0]))[0]
    r, g, b = (int(round(255 * c)) for c in rgba[:3])
    return f"#{r:02x}{g:02x}{b:02x}"


def _update_metric_labels(psf, px_nm, baseline_vals):
    """Refresh the metric Label widgets: text = value, color = deviation from
    the Airy baseline."""
    m = psf_metrics(psf, psz=(px_nm, px_nm))
    vals = _metric_values(psf, px_nm, m)
    for spec in METRIC_SPECS:
        lab = _metric_labels[spec["key"]]
        v, b = vals[spec["key"]], baseline_vals[spec["key"]]
        if spec["mode"] == "rel":
            dev = (v / b - 1.0) if b != 0 else 0.0
        else:
            dev = v - b
        color = _deviation_color(dev, spec["scale"])
        lab.value = f"{spec['name']}: {v:.4g}{spec.get('unit', '')}"
        lab.native.setStyleSheet(f"color: {color}; font-weight: bold;")


def compute(coeffs, na, wavelength_nm, pixel_size_nm, shape_dist=4.0):
    """Return (psf, pixel_size_nm, wavefront) for the given coefficients.

    ``psf`` is the unit-sum 2D PSF on a physical pixel grid (tier-3 API);
    ``wavefront`` is W(rho, theta) in waves on the pupil grid.
    """
    psf, px_nm = psf_2d_phys(
        coeffs,
        na=na,
        wavelength_nm=wavelength_nm,
        pixel_size_nm=pixel_size_nm,
        shape_dist=shape_dist,
    )
    w = wavefront(WAVEFRONT_N, coeffs, WAVEFRONT_RADIUS)
    return psf, float(px_nm), w


def compute_3d(coeffs, na, wavelength_nm, pixel_size_nm, dz_nm, shape_dist=4.0):
    """Return (stack, (dz_nm, dy_nm, dx_nm)) for the given coefficients.

    ``stack`` is the unit-sum-per-plane 3D PSF on a physical grid (tier-3
    API, paraxial defocus propagation between planes).
    """
    return psf_stack_phys(
        coeffs,
        na=na,
        wavelength_nm=wavelength_nm,
        pixel_size_nm=pixel_size_nm,
        dz_nm=dz_nm,
        shape_dist=shape_dist,
    )


def _update_layers(viewer, psf, px_nm, wavefront):
    """Create or update the PSF and Wavefront layers with the raw arrays.

    Only ``data`` and ``scale`` are assigned.  Contrast limits and gamma are
    intentionally left alone: as long as they haven't been set manually, napari
    keeps auto-contrast enabled and re-fits the contrast limits to the new data
    on every ``layer.data`` assignment, and the native ``layer.gamma`` slider
    (layer controls) stays usable.
    """
    if "PSF" in viewer.layers:
        layer = viewer.layers["PSF"]
        layer.data = psf
        layer.scale = (px_nm, px_nm)
    else:
        viewer.add_image(psf, name="PSF", colormap="magma", scale=(px_nm, px_nm))

    if "Wavefront" in viewer.layers:
        viewer.layers["Wavefront"].data = wavefront
    else:
        viewer.add_image(wavefront, name="Wavefront", colormap="RdBu")


# Build the widget. Per-parameter dicts in the decorator set the widget type
# and ranges; the signature gives names/defaults. auto_call=True re-runs the
# function on every slider change; the target viewer is taken from
# napari.current_viewer() inside the callback (robust to auto-calls, which
# re-invoke the function without the originally passed arguments).
_mode_options = {
    name: dict(
        widget_type="FloatSlider", min=lo, max=hi, step=0.01, value=0.0,
        label=f"{name} (waves)",
    )
    for name, (lo, hi) in MODE_SLIDERS.items()
}


@magicgui(
    layout="vertical",
    auto_call=True,
    call_button=False,
    **_mode_options,
    na=dict(widget_type="FloatSlider", min=0.5, max=1.4, step=0.01,
            value=1.3, label="NA"),
    wavelength_nm=dict(widget_type="FloatSlider", min=400.0, max=700.0,
                       step=5.0, value=520.0, label="wavelength (nm)"),
    pixel_size_nm=dict(widget_type="FloatSlider", min=20.0, max=200.0,
                       step=5.0, value=65.0, label="pixel size (nm)"),
    shape_dist=dict(widget_type="FloatSlider", min=2.0, max=8.0, step=0.5,
                    value=4.0, label="window (Airy diameters)"),
    dz_nm=dict(widget_type="FloatSlider", min=50.0, max=400.0, step=10.0,
               value=150.0, label="dz (nm)"),
    stack_3d=dict(widget_type="CheckBox", value=False, label="3D PSF stack"),
)
def zernike_widget(
    defocus: float = 0.0,
    astigmatism_0: float = 0.0,
    astigmatism_45: float = 0.0,
    coma_x: float = 0.0,
    coma_y: float = 0.0,
    trefoil_x: float = 0.0,
    spherical: float = 0.0,
    na: float = 1.3,
    wavelength_nm: float = 520.0,
    pixel_size_nm: float = 65.0,
    shape_dist: float = 4.0,
    dz_nm: float = 150.0,
    stack_3d: bool = False,
) -> None:
    coeffs = {
        name: value
        for name, value in [
            ("defocus", defocus),
            ("astigmatism_0", astigmatism_0),
            ("astigmatism_45", astigmatism_45),
            ("coma_x", coma_x),
            ("coma_y", coma_y),
            ("trefoil_x", trefoil_x),
            ("spherical", spherical),
        ]
        if value != 0.0
    }
    psf, px_nm, wavefront = compute(
        coeffs, na, wavelength_nm, pixel_size_nm, shape_dist
    )
    # Airy baseline metrics at the current optics (cached per setting)
    key = (round(na, 4), round(wavelength_nm, 1), round(pixel_size_nm, 1),
           round(shape_dist, 2))
    if key not in _baseline_cache:
        bpsf, bpx, _ = compute({}, na, wavelength_nm, pixel_size_nm, shape_dist)
        _baseline_cache[key] = _metric_values(
            bpsf, bpx, psf_metrics(bpsf, psz=(bpx, bpx)))
    _update_metric_labels(psf, px_nm, _baseline_cache[key])
    viewer = napari.current_viewer()
    if viewer is not None:
        _update_layers(viewer, psf, px_nm, wavefront)
        if stack_3d:
            st, (dz, dy, dx) = compute_3d(
                coeffs, na, wavelength_nm, pixel_size_nm, dz_nm, shape_dist
            )
            if "PSF3D" in viewer.layers:
                layer = viewer.layers["PSF3D"]
                layer.data = st
                layer.scale = (dz, dy, dx)
            else:
                viewer.add_image(st, name="PSF3D", colormap="magma",
                                 scale=(dz, dy, dx))
                viewer.dims.ndisplay = 3
        elif "PSF3D" in viewer.layers:
            viewer.layers.remove("PSF3D")
            viewer.dims.ndisplay = 2
    return psf  # handy for headless testing


# Reset button: restores all coefficient sliders to 0 (auto_call re-renders).
_reset_button = PushButton(text="Reset coefficients")


@_reset_button.clicked.connect
def _on_reset() -> None:
    # reset_choices() only covers choice-based widgets, so set sliders directly;
    # each assignment fires `changed` -> auto_call re-renders the layers
    for name in MODE_SLIDERS:
        getattr(zernike_widget, name).value = 0.0


zernike_widget.append(_reset_button)

# metric read-out labels, appended below the sliders / reset button
zernike_widget.append(Label(value="--- PSF metrics (color = deviation vs Airy) ---"))
_metric_labels = {}
for _spec in METRIC_SPECS:
    _lab = Label(value=f"{_spec['name']}: -")
    zernike_widget.append(_lab)
    _metric_labels[_spec["key"]] = _lab


def main() -> None:
    viewer = napari.Viewer()
    viewer.window.add_dock_widget(zernike_widget, name="Zernike PSF")
    zernike_widget()  # initial render with default (unaberrated) values
    napari.run()


if __name__ == "__main__":
    main()
