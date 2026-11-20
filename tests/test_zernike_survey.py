"""Smoke test for zernike_examples/zernike_metric_survey.py (no GUI, package
deps only).  Runs a tiny survey into a tmp dir and checks the outputs."""

import os
import sys

import numpy as np
import pytest

EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "zernike_examples")
sys.path.insert(0, os.path.abspath(EXAMPLES))

import zernike_metric_survey as survey  # noqa: E402


def test_synth_features_finite():
    f = survey.synth({"defocus": 0.2, "coma_x": 0.1})
    for feat in survey.FEATURES:
        assert feat in f
    assert f["peak"] > 0
    assert np.isfinite(f["fwhm_mean_nm"]) and f["fwhm_mean_nm"] > 0


def test_piston_does_not_change_psf():
    # sanity check of the synthesis itself: piston is a global phase
    f0 = survey.synth({})
    f1 = survey.synth({"piston": 0.5})
    assert f0["peak"] == pytest.approx(f1["peak"], rel=1e-9)
    assert f0["fwhm_mean_nm"] == pytest.approx(f1["fwhm_mean_nm"], rel=1e-9)


def test_tiny_survey_runs_and_writes_outputs(tmp_path):
    survey.main(["--amp-max", "0.1", "--n-amp", "3", "--n-random", "20",
                 "--outdir", str(tmp_path)])
    for name in ("metric_survey.png", "survey_oat.csv", "survey_random.csv",
                 "survey_sensitivity_abs.csv", "survey_sensitivity_signed.csv"):
        p = tmp_path / name
        assert p.exists() and p.stat().st_size > 0, name
    # PNG should not be blank: more than one distinct pixel value
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    img = plt.imread(tmp_path / "metric_survey.png")
    assert img[..., :3].std() > 0.01
