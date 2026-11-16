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

    # validation errors
    for bad in (
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
