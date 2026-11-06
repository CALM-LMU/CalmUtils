#!/usr/bin/env python3
"""Self-test for the generic restoration / PSF-metric functions moved from the
scratchbook PSF pipeline into calmutils:

  calmutils.restoration.subpixel_shift / rl_deconvolve / resample_psf /
      psf_metrics / profile_metrics / axis_profile / encircled_energy /
      radius_at_fraction
  calmutils.morphology.bead_kernel
  calmutils.localization.util.nearest_neighbor_distances

Synthetic data with known ground truth throughout. Run directly:

    $ python test_restoration.py

Prints per-section results and "TEST PASSED" / "TEST FAILED".
"""

import warnings

import numpy as np

from calmutils.restoration import (
    subpixel_shift, rl_deconvolve, resample_psf,
    axis_profile, profile_metrics, psf_metrics,
    encircled_energy, radius_at_fraction,
)
from calmutils.restoration.deconvolution import _convolve_same
from calmutils.morphology import bead_kernel
from calmutils.localization.util import (nearest_neighbor_distances,
                                         kernel_weights, effective_sample_size)


def _gaussian_volume(shape, sigma):
    grids = np.meshgrid(*[np.arange(s) - (s - 1) / 2.0 for s in shape],
                        indexing="ij")
    g = np.zeros(shape)
    for ax, s in zip(grids, sigma):
        g = g + (ax / s) ** 2
    return np.exp(-0.5 * g)


def _axis_sigma(psf, ax):
    m = psf.sum(axis=tuple(j for j in range(psf.ndim) if j != ax))
    c = len(m) // 2
    return np.sqrt((((np.arange(len(m)) - c) ** 2) * m).sum() / m.sum())


def main():
    ok = True

    # ---------------- subpixel_shift ----------------
    g = _gaussian_volume((41,), (3.0,))
    shifted = subpixel_shift(g, (0.37,))  # positive shift -> content moves to higher indices
    m = shifted
    com = (np.arange(len(m)) * m).sum() / m.sum()
    print(f"[shift] centroid after +0.37 shift: {com:.4f} (center 20 -> expect ~20.37)")
    ok &= abs(com - 20.37) < 0.01
    ok &= abs(shifted.sum() - g.sum()) < 1e-9  # shift preserves the sum

    # ---------------- _convolve_same alignment ----------------
    d = np.zeros((15,)); d[7] = 1.0
    k = _gaussian_volume((5,), (1.0,)); k /= k.sum()
    c = _convolve_same(d, k)
    print(f"[conv] delta at 7 -> peak at {np.argmax(c)}, sum {c.sum():.6f}")
    ok &= np.argmax(c) == 7 and abs(c.sum() - 1.0) < 1e-12

    # ---------------- bead_kernel ----------------
    tiny = bead_kernel((50.0, 50.0), 10.0)  # r = 0.1 px -> delta
    print(f"[bead_kernel] sub-voxel bead: shape {tiny.shape}, max {tiny.max():.3f}")
    ok &= tiny.shape == (3, 3) and tiny.max() == 1.0
    sph = bead_kernel((1.0, 1.0, 1.0), 2.0)  # r = 1 px
    print(f"[bead_kernel] r=1 sphere: shape {sph.shape}, sum {sph.sum():.6f}, "
          f"center {sph[1, 1, 1]:.3f}, symmetric: {np.allclose(sph, sph.transpose(2, 1, 0))}")
    ok &= sph.shape == (3, 3, 3) and abs(sph.sum() - 1.0) < 1e-12
    ok &= sph[1, 1, 1] == sph.max() and np.allclose(sph, sph.transpose(2, 1, 0))

    # ---------------- rl_deconvolve ----------------
    from skimage.restoration import richardson_lucy as sk_rl  # test oracle only
    from scipy.stats import chi2 as _chi2

    psz = (150.0, 50.0, 50.0)
    sigma_true = (3.0, 2.0, 2.0)
    k5 = bead_kernel((50.0, 50.0, 50.0), 250.0)  # r = 2.5 px -> 7^3 kernel
    g5 = _gaussian_volume((25, 21, 21), sigma_true)
    obs5 = _convolve_same(g5, k5)

    ours = rl_deconvolve(obs5, k5, num_iter=60)
    theirs = sk_rl(obs5, k5, num_iter=60, clip=False)
    rel = np.abs(ours - theirs).max() / theirs.max()
    flux = ours.sum() / obs5.sum()
    print(f"[rl] vs skimage: max rel diff {rel:.3f}, flux ratio {flux:.4f}")
    ok &= rel < 0.05 and abs(flux - 1.0) < 0.02
    for ax in range(3):
        s = _axis_sigma(ours / ours.sum(), ax)
        print(f"[rl] recovered sigma[{ax}]: {s:.3f} px, true {sigma_true[ax]} px")
        ok &= abs(s / sigma_true[ax] - 1) < 0.06

    # bandlimit (cycles/voxel) + damp
    ours_bl = rl_deconvolve(obs5, k5, num_iter=80, bandlimit=0.45 * np.ones(3))
    s_bl = [_axis_sigma(ours_bl / ours_bl.sum(), ax) for ax in range(3)]
    ours_d = rl_deconvolve(obs5, k5, num_iter=10, damp=1e-6 * obs5.max())
    print(f"[rl] bandlimit sigmas {np.round(s_bl, 3)} (true {sigma_true}); "
          f"damp min {ours_d.min():.1e}, flux {ours_d.sum() / obs5.sum():.4f}")
    ok &= all(abs(s_bl[ax] / sigma_true[ax] - 1) < 0.10 for ax in range(3))
    ok &= ours_d.min() >= 0.0 and abs(ours_d.sum() / obs5.sum() - 1.0) < 0.05

    # kernel guards: off-center warns, negative clipped+warns, even shape raises
    k_big = bead_kernel((50.0, 50.0, 50.0), 200.0)  # 5^3
    k_off = subpixel_shift(k_big, (0.3, 0.0, 0.0))
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        rl_deconvolve(obs5, k_off, num_iter=2)
    off_w = any("center of mass" in str(w.message) for w in wl)
    k_neg = k_big.copy(); k_neg.flat[0] -= 0.01
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        rl_deconvolve(obs5, k_neg, num_iter=2)
    neg_w = any("negative" in str(w.message) for w in wl)
    try:
        rl_deconvolve(obs5, np.ones((4, 4, 4)), num_iter=1)
        even_raises = False
    except ValueError:
        even_raises = True
    print(f"[rl] guards: off-center warns {off_w}, negative warns {neg_w}, "
          f"even kernel raises {even_raises}")
    ok &= off_w and neg_w and even_raises

    # edge="reflect": corner blob stays at the corner; periodic wraps it
    rng3 = np.random.default_rng(3)
    img_b = np.zeros((40, 40)); img_b[0, 0] = 100.0
    k_b = _gaussian_volume((9, 9), (2.0, 2.0)); k_b /= k_b.sum()
    obs_b = _convolve_same(np.pad(img_b, 20, mode="constant"), k_b)[20:-20, 20:-20]
    obs_b = obs_b + rng3.poisson(0.5, obs_b.shape)
    rec_r = rl_deconvolve(obs_b, k_b, num_iter=60, edge="reflect")
    rec_p = rl_deconvolve(obs_b, k_b, num_iter=60, edge="periodic")
    pk_r = np.unravel_index(np.argmax(rec_r), rec_r.shape)
    pk_p = np.unravel_index(np.argmax(rec_p), rec_p.shape)
    ghost_r, ghost_p = rec_r[33:, 33:].sum(), rec_p[33:, 33:].sum()
    print(f"[rl] reflect peak {pk_r} (true (0,0)); periodic peak {pk_p}; "
          f"far-corner mass {ghost_r:.2e} vs {ghost_p:.2e}; flux {rec_r.sum() / obs_b.sum():.3f}")
    ok &= abs(pk_r[0]) <= 1 and abs(pk_r[1]) <= 1
    ok &= ghost_p > ghost_r and 0.8 < rec_r.sum() / obs_b.sum() < 1.05

    # edge="reflect" in 3D with a non-cubic kernel (pad needs per-axis pairs)
    img3 = np.zeros((24, 30, 30))
    img3[2, 3, 3] = 100.0
    k3 = _gaussian_volume((5, 9, 9), (1.0, 2.0, 2.0)); k3 /= k3.sum()
    obs3 = _convolve_same(np.pad(img3, 40, mode="constant"), k3)[40:-40, 40:-40, 40:-40]
    rec3 = rl_deconvolve(obs3, k3, num_iter=40, edge="reflect")
    pk3 = np.unravel_index(np.argmax(rec3), rec3.shape)
    print(f"[rl] 3D reflect with non-cubic kernel: peak {pk3} (true (2,3,3)), "
          f"shape {rec3.shape} == input {obs3.shape}")
    ok &= rec3.shape == obs3.shape and abs(pk3[0] - 2) <= 1 and abs(pk3[1] - 3) <= 1

    # sub-voxel bead kernel -> deconvolution is a no-op
    g2 = _gaussian_volume((21, 25), (2.0, 3.0))
    noop = rl_deconvolve(g2, bead_kernel((50.0, 50.0), 10.0), num_iter=10)
    s_in = (_axis_sigma(g2, 0), _axis_sigma(g2, 1))
    s_out = (_axis_sigma(noop, 0), _axis_sigma(noop, 1))
    print(f"[rl] delta-kernel no-op sigmas {np.round(s_out, 3)} vs input {np.round(s_in, 3)}")
    ok &= abs(s_out[0] / s_in[0] - 1) < 0.02 and abs(s_out[1] / s_in[1] - 1) < 0.02

    # ---------------- resample_psf ----------------
    shape = (15, 21, 21)
    sigma = (1.5, 2.5, 2.5)
    psz_from = np.array([100.0, 40.0, 40.0])
    g = _gaussian_volume(shape, sigma)

    def expect_sigma(out, ax):
        # sigma in new px = sigma in old px * (n_new / n_old)
        return sigma[ax] * out.shape[ax] / shape[ax]

    up, up_psz = resample_psf(g, psz_from, psz_from / 2.0)
    print(f"[resample] up x2 -> shape {up.shape}")
    ok &= np.allclose(np.unravel_index(np.argmax(up), up.shape), np.array(up.shape) // 2)
    for ax in range(3):
        s = _axis_sigma(up, ax)
        ok &= abs(s / expect_sigma(up, ax) - 1) < 0.03
    ok &= abs(up.sum() - 1.0) < 1e-9

    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        down, _ = resample_psf(g, psz_from, psz_from * 2.0, out_size=(7, 11, 11))
    warned = any("deviates" in str(w.message) for w in wl)
    print(f"[resample] down x2 -> shape {down.shape}, mismatch warning fired: {warned}")
    ok &= warned
    ok &= np.allclose(np.unravel_index(np.argmax(down), down.shape), np.array(down.shape) // 2)
    for ax in range(3):
        ok &= abs(_axis_sigma(down, ax) / expect_sigma(down, ax) - 1) < 0.05

    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        mid, _ = resample_psf(g, psz_from, psz_from / 1.5)
    warned = any("deviates" in str(w.message) for w in wl)
    print(f"[resample] x1.5 -> shape {mid.shape}, mismatch warning fired: {warned}")
    ok &= warned
    for ax in range(3):
        ok &= abs(_axis_sigma(mid, ax) / expect_sigma(mid, ax) - 1) < 0.03

    back, _ = resample_psf(up, up_psz, psz_from, out_size=shape)
    for ax in range(3):
        s = _axis_sigma(back, ax)
        print(f"[resample] round-trip sigma[{ax}]: {s:.3f} px, original {sigma[ax]} px")
        ok &= abs(s / sigma[ax] - 1) < 0.05

    # ---------------- profile_metrics / psf_metrics ----------------
    xs = np.arange(41)
    prof = np.exp(-0.5 * ((xs - 20.3) / 2.5) ** 2)
    pm = profile_metrics(prof)
    fwhm_true = 2.5 * 2 * np.sqrt(2 * np.log(2))
    print(f"[profile] fwhm {pm['fwhm']:.4f} vs true {fwhm_true:.4f}, centroid "
          f"{pm['centroid']:.4f} vs 20.3, sigma {pm['sigma']:.4f} vs 2.5")
    ok &= abs(pm["fwhm"] / fwhm_true - 1) < 0.02
    ok &= abs(pm["centroid"] - 20.3) < 0.1
    ok &= abs(pm["sigma"] / 2.5 - 1) < 0.02 and not pm["truncated"]
    pm_t = profile_metrics(np.exp(-0.5 * ((xs - 20) / 20.0) ** 2))
    print(f"[profile] truncated flag on sigma=20 in 41 px window: {pm_t['truncated']}")
    ok &= pm_t["truncated"]

    g3 = _gaussian_volume((21, 25, 25), (2.0, 3.0, 3.0))
    g3 /= g3.sum()
    m3 = psf_metrics(g3, psz=(100.0, 50.0, 50.0))
    fw_true = np.array([2.0, 3.0, 3.0]) * 2 * np.sqrt(2 * np.log(2))
    print(f"[psf_metrics] fwhm_px {np.round(m3['fwhm_px'], 3)} vs {np.round(fw_true, 3)}, "
          f"peak_offset {np.round(m3['peak_offset_px'], 4)}, sidelobe "
          f"{m3['sidelobe_ratio']:.2e}, edge {m3['edge_level']:.2e}")
    ok &= np.allclose(m3["fwhm_px"], fw_true, rtol=0.02)
    ok &= np.allclose(m3["sigma_px"], [2, 3, 3], rtol=0.02)
    ok &= np.all(np.abs(m3["peak_offset_px"]) < 0.05)
    ok &= m3["sidelobe_ratio"] < 0.01 and m3["edge_level"] < 0.01
    ok &= np.allclose(m3["fwhm_phys"], np.array(m3["fwhm_px"]) * [100, 50, 50])

    g5v = _gaussian_volume((21, 25, 25), (1.5, 2.0, 2.0))
    g4 = g5v + 0.25 * (subpixel_shift(g5v, (0, 0, 7)) + subpixel_shift(g5v, (0, 0, -7)))
    g4 /= g4.sum()
    m4 = psf_metrics(g4)
    print(f"[psf_metrics] sidelobe ratio with 0.25 side peaks at +-7 px: {m4['sidelobe_ratio']:.3f}")
    ok &= 0.15 < m4["sidelobe_ratio"] < 0.4

    # ---------------- asymmetry metrics ----------------
    # asymmetric profile: Gaussian + tail to the right -> asymmetry > 0, skew > 0
    prof_a = np.exp(-0.5 * ((xs - 20) / 2.5) ** 2) \
        + 0.4 * np.exp(-0.5 * ((xs - 26) / 3.0) ** 2)
    pma = profile_metrics(prof_a)
    print(f"[asym] profile asymmetry {pma['asymmetry']:.3f} (expect > 0), "
          f"hw_left {pma['hw_left']:.2f} hw_right {pma['hw_right']:.2f}, "
          f"skew {pma['skew']:.3f} (expect > 0)")
    ok &= pma["asymmetry"] > 0.1 and pma["skew"] > 0.1
    ok &= abs(pma["hw_left"] + pma["hw_right"] - pma["fwhm"]) < 1e-9
    # symmetric Gaussian -> asymmetry ~ 0
    pms = profile_metrics(np.exp(-0.5 * ((xs - 20) / 2.5) ** 2))
    ok &= abs(pms["asymmetry"]) < 0.05 and abs(pms["skew"]) < 0.1

    # astigmatism: fwhm_x != fwhm_y
    g_ast = _gaussian_volume((21, 21, 31), (2.0, 2.0, 3.0))
    g_ast /= g_ast.sum()
    m_ast = psf_metrics(g_ast, psz=(100.0, 50.0, 50.0))
    exp_ast = (3.0 - 2.0) / 2.5  # (fwhm_x - fwhm_y) / mean, sigmas 2 vs 3
    print(f"[asym] astigmatism {m_ast['astigmatism']:.3f} vs expected ~{exp_ast:.3f}; "
          f"ring_radius_phys {np.round(m_ast['ring_radius_phys'], 1)} nm (no ring -> nan)")
    ok &= abs(m_ast["astigmatism"] - exp_ast) < 0.05
    ok &= np.all(np.isnan(m_ast["ring_radius_px"]))

    # donut: ring radius ~ +-7 px, symmetric sidelobes
    m4b = psf_metrics(g4, psz=(50.0, 50.0, 50.0))
    print(f"[asym] donut: sidelobe L/R {m4b['sidelobe_left']:.3f}/{m4b['sidelobe_right']:.3f}, "
          f"asym {m4b['sidelobe_asym']:.3f}, ring_radius_px {np.round(m4b['ring_radius_px'], 1)}")
    ok &= 5 < m4b["ring_radius_px"][2] < 9
    ok &= abs(m4b["sidelobe_asym"]) < 0.2
    ok &= abs(m4b["sidelobe_left"] - m4b["sidelobe_right"]) < 0.05

    # ---------------- kernel weights / ESS ----------------
    pts_k = np.array([[0., 0.], [3., 0.], [0., 4.], [30., 30.]])
    w = kernel_weights(pts_k, [0., 0.], bandwidth=5.0)
    print(f"[kernel] weights {np.round(w, 3)} (center 1, decreasing)")
    ok &= w[0] == 1.0 and w[1] > w[2] > w[3] and w[3] < 1e-8
    w_phys = kernel_weights(pts_k, [0., 0.], bandwidth=10.0, psz=(2.0, 1.0))
    # physical distance of p1 = 6 > bandwidth -> smaller weight than p2 (dist 4)
    ok &= w_phys[2] > w_phys[1]
    w_ep = kernel_weights(pts_k, [0., 0.], bandwidth=5.0, kind="epanechnikov")
    ok &= w_ep[3] == 0.0 and np.all(w_ep >= 0)
    ess_uni = effective_sample_size(np.ones(10))
    ess_peak = effective_sample_size(np.array([1.0] + [1e-6] * 9))
    print(f"[kernel] ESS uniform {ess_uni:.1f} (expect 10), peaked {ess_peak:.2f} (expect ~1)")
    ok &= abs(ess_uni - 10) < 1e-9 and abs(ess_peak - 1) < 1e-3

    # ---------------- encircled energy ----------------
    g_iso = _gaussian_volume((41, 41, 41), (2.0, 2.0, 2.0))
    g_iso /= g_iso.sum()
    psz_iso = (50.0, 50.0, 50.0)
    ee1 = encircled_energy(g_iso, 1 * 2.0 * 50.0, psz_iso)
    ee2 = encircled_energy(g_iso, 2 * 2.0 * 50.0, psz_iso)
    r80 = radius_at_fraction(g_iso, 0.8, psz_iso)
    exp1, exp2 = _chi2.cdf(1.0, 3), _chi2.cdf(4.0, 3)
    exp_r80 = 2.0 * 50.0 * np.sqrt(_chi2.ppf(0.8, 3))
    print(f"[ee] EE(1 sigma) {ee1:.4f} vs {exp1:.4f}; EE(2 sigma) {ee2:.4f} vs "
          f"{exp2:.4f}; r80 {r80:.1f} nm vs {exp_r80:.1f} nm")
    ok &= abs(ee1 - exp1) < 0.02 and abs(ee2 - exp2) < 0.02
    ok &= abs(r80 / exp_r80 - 1) < 0.05

    # ---------------- nearest_neighbor_distances ----------------
    pts = np.array([[0., 0.], [3., 0.], [0., 4.], [10., 10.]])
    nn = nearest_neighbor_distances(pts)
    nn_exp = np.array([3.0, 3.0, 4.0, np.sqrt(136.0)])
    print(f"[nn] distances (voxels): {np.round(nn, 3)} vs {np.round(nn_exp, 3)}")
    ok &= np.allclose(nn, nn_exp)
    nn_phys = nearest_neighbor_distances(pts, psz=(2.0, 1.0))
    nn_phys_exp = np.array([4.0, 6.0, 4.0, np.sqrt(296.0)])
    print(f"[nn] distances (psz=(2,1)): {np.round(nn_phys, 3)} vs {np.round(nn_phys_exp, 3)}")
    ok &= np.allclose(nn_phys, nn_phys_exp)
    nn_dup = nearest_neighbor_distances(np.array([[1., 1.], [1., 1.], [5., 5.]]))
    nn_one = nearest_neighbor_distances(np.array([[1., 1.]]))
    print(f"[nn] duplicates -> {nn_dup[:2]}, single point -> {nn_one}")
    ok &= nn_dup[0] == 0.0 and nn_dup[1] == 0.0 and np.isinf(nn_one[0])

    print("TEST", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    main()
