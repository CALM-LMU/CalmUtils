"""Systematic survey: how do the PSF metrics respond to Zernike coefficients?

Exploratory tool (not part of the package).  Goal: find redundancies (metrics
that move together) and gaps (modes no metric sees, or aspects no metric
quantifies) in ``calmutils.restoration.psf_metrics``.

Design
------
1. One-at-a-time (OAT) sweeps: for each mode, amplitude in
   ``linspace(-amp_max, amp_max, n_amp)`` (default +-0.2 waves - realistic
   aberrations; use ``--amp-max 1.0`` for the drastic regime).  From each sweep
   we get the *sensitivity* of every metric to that mode: least-squares slope
   vs |amp| (even responses, e.g. defocus) and vs amp (odd responses, e.g.
   coma/tilt), normalized per metric by the random-design std.
2. Random design: ``n_random`` coefficient vectors drawn uniformly in
   [-amp_max, amp_max]^n_modes.  Used for the metric correlation matrix
   (Spearman, monotone redundancy) and a PCA of the standardized metric space
   (effective dimensionality vs. number of modes).

Outputs (in ``--outdir``, default: ``output/`` next to this script):
- ``metric_survey.png`` : 3 panels - metric correlation heatmap, mode x metric
  sensitivity heatmap (percentile ranks), PCA scree.
- ``survey_oat.csv`` / ``survey_random.csv`` : raw samples (one row per PSF);
  load a row's coefficients in zernike_psf_viewer.py to verify interactively.
- ``survey_sensitivity_abs.csv`` / ``survey_sensitivity_signed.csv`` :
  normalized sensitivity matrices.

Run (needs only numpy/scipy/matplotlib, all package deps):
    python exploration/zernike/zernike_metric_survey.py --amp-max 0.2 --n-amp 9 --n-random 300
"""

import argparse
import csv
import os

import numpy as np
from scipy.stats import spearmanr, rankdata

from calmutils.restoration.zernike import psf_2d_phys
from calmutils.restoration.psf_metrics import psf_metrics

# optics fixed to the viewer defaults (zernike_psf_viewer.py)
NA, WAVELENGTH_NM, PIXEL_NM = 1.3, 520.0, 65.0
SHAPE_DIST = 4.0

# modes swept one-at-a-time; piston/tilt are sanity & gap checks (piston must
# not change the PSF at all; tilt shifts it - only offset metrics should move)
MODES = [
    "defocus", "astigmatism_0", "astigmatism_45", "coma_x", "coma_y",
    "trefoil_x", "spherical", "tilt_x", "tilt_y", "piston",
]

# metric feature vector extracted from psf_metrics() (+ peak); all scalars
FEATURES = [
    "peak", "fwhm_mean_nm", "sigma_mean_nm", "sidelobe_ratio",
    "sidelobe_asym", "asym_x", "asym_y", "skew_x", "skew_y", "astigmatism",
    "elongation", "orientation_deg", "offset_x_nm", "offset_y_nm",
    "ring_radius_mean_nm", "edge_level",
]


def features_of(psf, px_nm):
    m = psf_metrics(psf, psz=(px_nm, px_nm))
    astig = m["astigmatism"]
    off = m["peak_offset_phys"]
    ring = m["ring_radius_phys"]
    return {
        "peak": float(psf.max()),
        "fwhm_mean_nm": float(np.mean(m["fwhm_phys"])),
        "sigma_mean_nm": float(np.mean(m["sigma_px"])) * px_nm,
        "sidelobe_ratio": float(m["sidelobe_ratio"]),
        "sidelobe_asym": float(m["sidelobe_asym"]),
        "asym_x": float(m["asymmetry_px"][1]),
        "asym_y": float(m["asymmetry_px"][0]),
        "skew_x": float(m["skew"][1]),
        "skew_y": float(m["skew"][0]),
        "astigmatism": float(astig) if astig is not None else np.nan,
        "elongation": float(m["elongation"]),
        "orientation_deg": float(m["orientation_deg"]),
        "offset_x_nm": float(off[1]),
        "offset_y_nm": float(off[0]),
        "ring_radius_mean_nm": float(np.nanmean(ring)) if np.isfinite(ring).any() else np.nan,
        "edge_level": float(m["edge_level"]),
    }


def synth(coeffs):
    """PSF at the fixed survey optics -> feature dict."""
    psf, px = psf_2d_phys(coeffs, na=NA, wavelength_nm=WAVELENGTH_NM,
                         pixel_size_nm=PIXEL_NM, shape_dist=SHAPE_DIST)
    return features_of(psf, px)


def write_csv(path, rows):
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def matrix_to_csv(path, mat, row_labels, col_labels):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([""] + col_labels)
        for lab, row in zip(row_labels, mat):
            w.writerow([lab] + [f"{v:.6g}" for v in row])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--amp-max", type=float, default=0.2,
                    help="sweep/random range in waves, +- this value (default 0.2)")
    ap.add_argument("--n-amp", type=int, default=9, help="OAT sweep points per mode")
    ap.add_argument("--n-random", type=int, default=300, help="random design size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "output"))
    args = ap.parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    n_modes, n_feat = len(MODES), len(FEATURES)

    # ---- 1. OAT sweeps ------------------------------------------------------
    amps = np.linspace(-args.amp_max, args.amp_max, args.n_amp)
    oat_rows = []
    for mode in MODES:
        for a in amps:
            f = synth({mode: float(a)})
            f["mode"], f["amp"] = mode, float(a)
            oat_rows.append(f)
    write_csv(os.path.join(args.outdir, "survey_oat.csv"), oat_rows)
    oat_amp = {m: np.array([r["amp"] for r in oat_rows if r["mode"] == m])
               for m in MODES}
    oat_X = {m: np.array([[r[f] for f in FEATURES] for r in oat_rows
                          if r["mode"] == m]) for m in MODES}

    # sensitivity per mode, two flavors:
    #  sens_abs = LSQ slope of metric vs |amp| (even responses: defocus, ...)
    #  sens_lin = LSQ slope of metric vs amp   (odd responses: coma, tilt, ...)
    sens_abs = np.zeros((n_modes, n_feat))
    sens_lin = np.zeros((n_modes, n_feat))
    for i, mode in enumerate(MODES):
        for j in range(n_feat):
            sens_abs[i, j] = np.polyfit(np.abs(oat_amp[mode]), oat_X[mode][:, j], 1)[0]
            sens_lin[i, j] = np.polyfit(oat_amp[mode], oat_X[mode][:, j], 1)[0]

    # ---- 2. random design ---------------------------------------------------
    rand_rows = []
    for _ in range(args.n_random):
        c = rng.uniform(-args.amp_max, args.amp_max, n_modes)
        coeffs = dict(zip(MODES, (float(v) for v in c)))
        f = synth(coeffs)
        f.update(coeffs)
        rand_rows.append(f)
    write_csv(os.path.join(args.outdir, "survey_random.csv"), rand_rows)

    X = np.array([[r[f] for f in FEATURES] for r in rand_rows], dtype=float)
    X = np.where(np.isfinite(X), X, np.nanmedian(X, axis=0))
    scale = X.std(axis=0)
    scale[scale == 0] = 1.0

    # metric redundancy: Spearman correlation over the random design
    corr = spearmanr(X).statistic

    # normalize sensitivities per metric by the random-design std
    sens_n = sens_abs / scale[None, :]
    sens_n_lin = sens_lin / scale[None, :]
    matrix_to_csv(os.path.join(args.outdir, "survey_sensitivity_abs.csv"),
                  np.abs(sens_n), MODES, FEATURES)
    matrix_to_csv(os.path.join(args.outdir, "survey_sensitivity_signed.csv"),
                  sens_n_lin, MODES, FEATURES)

    # PCA of the standardized metric space
    Xz = (X - X.mean(axis=0)) / scale
    _, S, Vt = np.linalg.svd(Xz, full_matrices=False)
    var_expl = S ** 2 / (S ** 2).sum()

    # ---- 3. figure (compact, 3 panels) --------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(19, 6.5))
    ax1 = fig.add_subplot(131)
    im = ax1.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax1.set_xticks(range(n_feat), FEATURES, rotation=90, fontsize=7)
    ax1.set_yticks(range(n_feat), FEATURES, fontsize=7)
    ax1.set_title(f"metric Spearman corr (random design, ±{args.amp_max} w)")
    fig.colorbar(im, ax=ax1, shrink=0.8)

    ax2 = fig.add_subplot(132)
    # percentile-rank colormap: a few mode x metric combos dominate in absolute
    # terms, so any magnitude-based scale (linear, log, gamma) crushes or bands
    # the rest.  Ranking all cells gives a uniform color histogram; absolute
    # normalized sensitivities are in survey_sensitivity_abs.csv.  Exact zeros
    # (piston row) and NaN (e.g. ring_radius for Airy-like PSFs) stay at rank 0.
    sn = np.abs(np.nan_to_num(sens_n, nan=0.0))
    ranks = rankdata(sn.ravel(), method="average").reshape(sn.shape) / sn.size
    im = ax2.imshow(ranks, cmap="magma", aspect="auto", vmin=0.0, vmax=1.0)
    ax2.set_xticks(range(n_feat), FEATURES, rotation=90, fontsize=7)
    ax2.set_yticks(range(n_modes), MODES, fontsize=8)
    ax2.set_title("|sensitivity| d(metric)/d|amp|, normalized (percentile rank)")
    cb = fig.colorbar(im, ax=ax2, shrink=0.8, ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
    cb.set_ticklabels(["min", "25%", "50%", "75%", "max"])
    cb.set_label("percentile of normalized |sens|")

    ax3 = fig.add_subplot(133)
    ax3.bar(range(1, len(var_expl) + 1), var_expl * 100)
    ax3.set_xlabel("PC")
    ax3.set_ylabel("variance explained (%)")
    ax3.set_title(f"PCA of {n_feat} metrics over {n_modes} modes "
                  f"({args.n_random} random PSFs)")
    fig.tight_layout()
    fig_path = os.path.join(args.outdir, "metric_survey.png")
    fig.savefig(fig_path, dpi=130)
    print(f"saved {fig_path}")

    # ---- 4. text summary ----------------------------------------------------
    print(f"\n=== survey: ±{args.amp_max} waves, {args.n_amp} OAT pts/mode, "
          f"{args.n_random} random PSFs, optics NA={NA} wl={WAVELENGTH_NM} px={PIXEL_NM}")

    print("\n-- most redundant metric pairs (top 8 |Spearman rho|, random design) --")
    pairs = sorted(((abs(corr[i, j]), FEATURES[i], FEATURES[j], corr[i, j])
                    for i in range(n_feat) for j in range(i + 1, n_feat)),
                   reverse=True)[:8]
    for _, a, b, r in pairs:
        flag = "  <-- redundant" if abs(r) > 0.9 else ""
        print(f"  {a:20s} vs {b:20s} rho={r:+.3f}{flag}")

    print("\n-- top responding metrics per mode (normalized |sens| vs |amp|) --")
    for i, mode in enumerate(MODES):
        top = np.argsort(-np.abs(sens_n[i]))[:4]
        print(f"  {mode:18s}: " + ", ".join(f"{FEATURES[j]}({sens_n[i, j]:.2f})"
                                            for j in top))

    print("\n-- strongest signed (linear, odd) responses --")
    for i, mode in enumerate(MODES):
        j = int(np.argmax(np.abs(sens_n_lin[i])))
        if abs(sens_n_lin[i, j]) > 0.3:
            print(f"  {mode:18s}: {FEATURES[j]} slope {sens_n_lin[i, j]:+.2f} per wave")

    # a mode/metric "responds" if either the even (|amp|) or odd (signed)
    # sensitivity is appreciable
    resp_mode = np.maximum(np.abs(sens_n).max(axis=1), np.abs(sens_n_lin).max(axis=1))
    resp_feat = np.maximum(np.abs(sens_n).max(axis=0), np.abs(sens_n_lin).max(axis=0))

    print("\n-- mode blind spots: modes whose max normalized sensitivity < 0.3 --")
    for i, mode in enumerate(MODES):
        if resp_mode[i] < 0.3:
            print(f"  {mode:20s} max |sens| = {resp_mode[i]:.3f}  <-- barely any metric moves")

    print("\n-- metric gaps: metrics with near-zero sensitivity to ALL modes "
          "(max |sens| < 0.05) --")
    gaps = [(FEATURES[j], resp_feat[j]) for j in range(n_feat) if resp_feat[j] < 0.05]
    for feat, v in gaps:
        print(f"  {feat:20s} max |sens| = {v:.3f}  <-- metric never moves in OAT")
    if not gaps:
        print("  (none)")

    print("\n-- PCA: variance explained by first PCs --")
    for i in range(min(6, len(var_expl))):
        top = np.argsort(-np.abs(Vt[i]))[:3]
        print(f"  PC{i+1}: {var_expl[i]*100:5.1f}%  top loadings: "
              + ", ".join(f"{FEATURES[j]}({Vt[i, j]:+.2f})" for j in top))
    n95 = int((np.cumsum(var_expl) < 0.95).sum()) + 1
    print(f"  -> {n_feat} metrics span ~{n95} PCs at 95% variance over {n_modes} modes")


if __name__ == "__main__":
    main()
