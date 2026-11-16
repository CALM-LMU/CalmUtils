#!/usr/bin/env python3
"""Self-test for calmutils.restoration.rl_deconvolve_multiview (moved from the
scratchbook multi-PSF work).

Synthetic two-view experiment with known ground truth: one object seen through
two complementary PSFs (A: sharp lateral / elongated axial, B: isotropic but
coarser). Checks that the joint estimate is sharper per axis than any single
view, that sum is approximately conserved, and that input validation works.

Run directly:

    $ python test_restoration_multiview.py

Prints per-section results and "TEST PASSED" / "TEST FAILED".
"""
import numpy as np

from calmutils.restoration import rl_deconvolve, rl_deconvolve_multiview


def _gauss(shape, sigmas):
    grids = np.meshgrid(*[np.arange(s) - (s - 1) / 2.0 for s in shape], indexing="ij")
    g = np.zeros(shape)
    for ax, s in zip(grids, sigmas):
        g = g + (ax / s) ** 2
    g = np.exp(-0.5 * g)
    return g / g.sum()


def _fwhm(p):
    p = np.asarray(p, float)
    p = p / p.max()
    i = int(np.argmax(p))
    lo = hi = i
    while lo > 0 and p[lo - 1] >= 0.5:
        lo -= 1
    while hi < len(p) - 1 and p[hi + 1] >= 0.5:
        hi += 1
    lo_x = lo - (0.5 - p[lo - 1]) / (p[lo] - p[lo - 1]) if lo > 0 else float(lo)
    hi_x = hi + (0.5 - p[hi]) / (p[hi + 1] - p[hi]) if hi < len(p) - 1 else float(hi)
    return hi_x - lo_x


def _bead_fwhm(vol, y, x, x_stop):
    """2D: FWHM along axis 1 (x) at row y, restricted to columns < x_stop so
    the profile excludes the stripes, and along axis 0 (z) at column x."""
    return _fwhm(vol[y, :x_stop]), _fwhm(vol[:, x])


def main():
    ok = True
    rng = np.random.default_rng(42)
    N, K, BG, N_ITER = 128, 65, 2.0, 200
    y, x, x_stop = 64, 55, 65                 # bead position; stripes start at x=70
    truth = np.zeros((N, N))
    truth[y, x] = 1.0                         # bead (away from stripes)
    truth[40:90, 70:73] = 0.1                 # fine stripe pair
    truth[40:90, 76:79] = 0.1

    # sigmas are per axis: (axis 0 = z, axis 1 = x)
    psf_a = _gauss((K, K), (8.0, 1.5))        # view A: sharp x, elongated z
    psf_b = _gauss((K, K), (4.0, 4.0))        # view B: isotropic, coarser

    def acquire(psf, peak):
        fs = [N + K - 1] * 2
        ax = (0, 1)
        clean = np.fft.irfftn(np.fft.rfftn(truth, fs, axes=ax)
                              * np.fft.rfftn(psf, fs, axes=ax), fs, axes=ax)
        clean = np.maximum(clean[K // 2:K // 2 + N, K // 2:K // 2 + N], 0.0)
        clean = clean * (peak / clean[y, x])
        return rng.poisson(clean + BG).astype(float)

    d_a = acquire(psf_a, 120)                 # dim view
    d_b = acquire(psf_b, 500)                 # bright view (~5x the flux of A)

    fx_a, fz_a = _bead_fwhm(d_a - BG, y, x, x_stop)
    fx_b, fz_b = _bead_fwhm(d_b - BG, y, x, x_stop)
    print(f"[raw]   view A x={fx_a:.2f} z={fz_a:.2f} | view B x={fx_b:.2f} z={fz_b:.2f}")

    s_a = rl_deconvolve(d_a, psf_a, num_iter=N_ITER, bg=BG)
    s_b = rl_deconvolve(d_b, psf_b, num_iter=N_ITER, bg=BG)
    j = rl_deconvolve_multiview([d_a, d_b], [psf_a, psf_b], num_iter=N_ITER, bg=BG)
    jx_a, jz_a = _bead_fwhm(s_a, y, x, x_stop)
    jx_b, jz_b = _bead_fwhm(s_b, y, x, x_stop)
    jx, jz = _bead_fwhm(j, y, x, x_stop)
    print(f"[RL]    A-only x={jx_a:.2f} z={jz_a:.2f} | B-only x={jx_b:.2f} z={jz_b:.2f} "
          f"| joint x={jx:.2f} z={jz:.2f}")

    # joint = best of both: as sharp as the better single-view result per axis
    # and clearly sharper than the worse one
    ok &= jx < jx_a + 0.5 and jx < jx_b - 1.0
    ok &= jz < jz_b + 0.5 and jz < jz_a - 1.0
    # flux: output is in units of an average-brightness view, bg-free
    frac = j.sum() / np.mean([(d_a - BG).sum(), (d_b - BG).sum()])
    print(f"[flux]  joint sum / mean bg-subtracted view sum = {frac:.3f}")
    ok &= 0.8 < frac < 1.2
    # explicit view_scales (true flux ratio) agree with the automatic estimate
    fl = [(d_a - BG).sum(), (d_b - BG).sum()]
    j2 = rl_deconvolve_multiview([d_a, d_b], [psf_a, psf_b], num_iter=N_ITER, bg=BG,
                                 view_scales=fl)
    ok &= np.allclose(j, j2, rtol=1e-6, atol=1e-9)
    print(f"[scale] explicit view_scales match auto estimate: {np.allclose(j, j2, rtol=1e-6, atol=1e-9)}")
    # non-negativity and shape
    ok &= j.shape == d_a.shape and (j >= 0).all()

    # blurred-avg init runs
    s = rl_deconvolve_multiview([d_a, d_b], [psf_a, psf_b], num_iter=3, bg=2.0)
    ok &= s.shape == d_a.shape and (s >= 0).all()
    print("[init]  blurred-avg init runs")

    # per-view weights: strips without data (as after registration) are
    # ignored, whatever values they hold; the other view fills them in
    w_a = np.ones((N, N)); w_a[:, :6] = 0.0
    w_b = np.ones((N, N)); w_b[:5, :] = 0.0
    g_a = d_a.copy(); g_a[:, :6] = np.nan
    g_b = d_b.copy(); g_b[:5, :] = 1e4
    jw = rl_deconvolve_multiview([g_a, g_b], [psf_a, psf_b], num_iter=N_ITER, bg=BG,
                                 weights=[w_a, w_b])
    wx, wz = _bead_fwhm(jw, y, x, x_stop)
    w_flux = jw.sum() / j.sum()
    print(f"[weights] masked strips: joint x={wx:.2f} z={wz:.2f}, flux vs unmasked {w_flux:.3f}")
    ok &= bool(np.isfinite(jw).all()) and abs(wx - jx) < 0.3 and abs(wz - jz) < 0.3
    ok &= abs(w_flux - 1.0) < 0.05

    # padding="extend": same result for an object inside the window ...
    je = rl_deconvolve_multiview([d_a, d_b], [psf_a, psf_b], num_iter=N_ITER, bg=BG,
                                 padding="extend")
    ex, ez = _bead_fwhm(je, y, x, x_stop)
    print(f"[extend] joint x={ex:.2f} z={ez:.2f}, shape {je.shape}")
    ok &= je.shape == d_a.shape and abs(ex - jx) < 0.3 and abs(ez - jz) < 0.3
    # ... and light from a source just outside the window is attributed to the
    # margin instead of the edge voxels (window truth is empty here)
    big = np.zeros((N + 2 * K, N + 2 * K))
    big[K - 4, K + 60] = 2e4                  # 4 px above the window
    fs = [N + 3 * K - 1] * 2
    blur = np.fft.irfftn(np.fft.rfftn(big, fs, axes=(0, 1))
                         * np.fft.rfftn(psf_b, fs, axes=(0, 1)), fs, axes=(0, 1))
    blur = np.maximum(blur[K // 2:K // 2 + N + 2 * K, K // 2:K // 2 + N + 2 * K], 0.0)
    obs = rng.poisson(blur[K:K + N, K:K + N] + BG).astype(float)
    r_n = rl_deconvolve(obs, psf_b, num_iter=100, bg=BG)
    r_e = rl_deconvolve(obs, psf_b, num_iter=100, bg=BG, padding="extend")
    print(f"[extend] outside source: in-window sum, zero border {r_n.sum():.0f} "
          f"vs extend {r_e.sum():.0f}")
    ok &= r_e.sum() < 0.3 * r_n.sum()

    # validation errors
    for bad in (
        lambda: rl_deconvolve_multiview([d_a, d_b], [psf_a, psf_b], weights=[w_a]),           # weights len
        lambda: rl_deconvolve_multiview([d_a, d_b], [psf_a, psf_b], weights=-w_a),            # weights < 0
        lambda: rl_deconvolve_multiview([d_a, d_b], [psf_a, psf_b], padding="nope"),          # padding
        lambda: rl_deconvolve_multiview([d_a, d_b], [psf_a]),                      # len mismatch
        lambda: rl_deconvolve_multiview([d_a, d_b[:, :-1]], [psf_a, psf_b]),        # shape mismatch
        lambda: rl_deconvolve_multiview([d_a], [np.ones((3, 3, 3))]),                 # kernel ndim mismatch
        lambda: rl_deconvolve_multiview([d_a], [psf_a], bg=[1.0, 2.0]),             # bg len
        lambda: rl_deconvolve_multiview([d_a, d_b], [psf_a, psf_b], view_scales=[1.0]),       # scales len
        lambda: rl_deconvolve_multiview([d_a, d_b], [psf_a, psf_b], view_scales=[1.0, 0.0]),  # scales <= 0
    ):
        try:
            bad()
            ok = False
            print("[validate] MISSING expected error")
        except (ValueError, Warning) as e:
            pass
    print("[validate] input validation raises as expected")

    print("TEST PASSED" if ok else "TEST FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
