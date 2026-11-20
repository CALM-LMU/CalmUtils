"""Smoke test for zernike_examples/zernike_psf_viewer.py.

Needs the ``viz`` extra (napari/magicgui); skipped otherwise.  Runs offscreen
(Qt) with a stub viewer - no GL canvas - so it works headless in CI.
"""

import os
import sys

import numpy as np
import pytest

pytest.importorskip("napari")
pytest.importorskip("magicgui")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from qtpy.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "exploration", "zernike")
sys.path.insert(0, os.path.abspath(EXAMPLES))

import napari  # noqa: E402
import zernike_psf_viewer as zv  # noqa: E402


class _StubLayer:
    def __init__(self, data, **kw):
        self.data, self.kw = data, kw
        self.scale = kw.get("scale")


class _StubDims:
    ndisplay = 2


class _StubLayers(dict):
    def remove(self, name):
        del self[name]


class _StubViewer:
    def __init__(self):
        self.layers = _StubLayers()
        self.dims = _StubDims()

    def add_image(self, data, name=None, **kw):
        self.layers[name] = _StubLayer(data, **kw)


def test_compute_2d_and_3d():
    psf, px, w = zv.compute({"coma_x": 0.3}, 1.3, 520.0, 65.0)
    assert abs(psf.sum() - 1) < 1e-9 and px == pytest.approx(65.0)
    assert np.abs(w).max() > 0.2
    st, (dz, dy, dx) = zv.compute_3d({"defocus": 0.3}, 1.3, 520.0, 65.0, 150.0)
    assert st.ndim == 3
    assert np.allclose(st.sum(axis=(1, 2)), 1.0)
    assert dz == pytest.approx(150.0, rel=0.05) and dy == pytest.approx(65.0, rel=0.05)


def test_widget_updates_layers_and_metrics(monkeypatch):
    viewer = _StubViewer()
    monkeypatch.setattr(napari, "current_viewer", lambda: viewer)

    # drive the GUI function with explicit kwargs (setting Qt slider .value
    # aborts under the napari pytest plugin's Qt setup)
    zv.zernike_widget()
    assert sorted(viewer.layers) == ["PSF", "Wavefront"]
    psf_layer = viewer.layers["PSF"]
    assert psf_layer.scale == (65.0, 65.0)
    assert abs(psf_layer.data.sum() - 1) < 1e-9  # raw PSF, no display transform
    assert "contrast_limits" not in psf_layer.kw  # we never touch contrast
    # unaberrated -> all metric labels near-white (zero deviation)
    for lab in zv._metric_labels.values():
        assert "color: #" in lab.native.styleSheet()

    peak_text0 = zv._metric_labels["peak"].value
    zv.zernike_widget(defocus=1.0)
    assert abs(viewer.layers["PSF"].data.sum() - 1) < 1e-9
    # metric labels re-rendered: peak (Strehl) dropped and changed color
    peak_lab = zv._metric_labels["peak"]
    assert peak_lab.value != peak_text0
    assert "color: #" in peak_lab.native.styleSheet()

    # 3D stack layer appears / disappears with the checkbox
    assert "PSF3D" not in viewer.layers
    zv.zernike_widget(defocus=1.0, stack_3d=True)
    assert "PSF3D" in viewer.layers and viewer.layers["PSF3D"].data.ndim == 3
    assert viewer.dims.ndisplay == 3
    zv.zernike_widget()
    assert "PSF3D" not in viewer.layers and viewer.dims.ndisplay == 2

    # back to unaberrated -> peak label returns to its baseline text
    assert zv._metric_labels["peak"].value == peak_text0
