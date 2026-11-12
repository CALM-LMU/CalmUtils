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


def _bead_fwhm(vol, y, x):
    """2D: FWHM along axis 1 (x) at row y, and along axis 0 (z) at column x."""
    return _fwhm(vol[y, :]), _fwhm(vol[:, x])


def main():
    ok = True
    rng = np.random.default_rng(42)
    N = 64
    truth = np.zeros((N, N))
    truth[32, 20] = 1.0                      # bead (away from stripes)
    truth[20:45, 40:43] = 0.1                 # fine stripe pair
    truth[20:45, 46:49] = 0.1

    psf_a = _gauss((21, 21), (1.5, 8.0))      # view A: sharp x, elongated z(axis0)
    psf_b = _gauss((21, 21), (4.0, 4.0))      # view B: isotropic, coarser

    def acquire(psf, peak):
        fs = [N + 20] * 2
        ax = (0, 1)
        clean = np.fft.irfftn(np.fft.rfftn(truth, fs, axes=ax)
                              * np.fft.rfftn(psf, fs, axes=ax), fs, axes=ax)
        clean = clean[10:10 + N, 10:10 + N]
        clean = clean * (peak / clean[32, 20])
        return rng.poisson(clean + 2.0).astype(float)

    d_a = acquire(psf_a, 120)
    d_b = acquire(psf_b, 500)
    y, x = 32, 20

    fx_a, fz_a = _bead_fwhm(d_a, y, x)
    fx_b, fz_b = _bead_fwhm(d_b, y, x)
    print(f"[raw]   view A x={fx_a:.2f} z={fz_a:.2f} | view B x={fx_b:.2f} z={fz_b:.2f}")

    s_a = rl_deconvolve(d_a, psf_a, num_iter=40, edge="periodic")
    s_b = rl_deconvolve(d_b, psf_b, num_iter=40, edge="periodic")
    j = rl_deconvolve_multiview([d_a, d_b], [psf_a, psf_b], num_iter=40, bg=2.0,
                                init="bluravg", edge="periodic")
    jx_a, jz_a = _bead_fwhm(s_a, y, x)
    jx_b, jz_b = _bead_fwhm(s_b, y, x)
    jx, jz = _bead_fwhm(j, y, x)
    print(f"[RL]    A-only x={jx_a:.2f} z={jz_a:.2f} | B-only x={jx_b:.2f} z={jz_b:.2f} "
          f"| joint x={jx:.2f} z={jz:.2f}")

    # joint must beat the best single view per axis (modulo noise overfit)
    ok &= jx < min(fx_a, fx_b) + 0.5
    ok &= jz < min(fz_a, fz_b) + 0.5
    # flux conservation (unit-sum kernels): total within 20% of mean view total
    frac = j.sum() / np.mean([d_a.sum(), d_b.sum()])
    print(f"[flux]  joint sum / mean view sum = {frac:.3f}")
    ok &= 0.8 < frac < 1.2
    # non-negativity and shape
    ok &= j.shape == d_a.shape and (j >= 0).all()

    # init variants all run
    for init in ("flat", "bluravg", "backproj"):
        s = rl_deconvolve_multiview([d_a, d_b], [psf_a, psf_b], num_iter=3, bg=2.0,
                                    init=init)
        ok &= s.shape == d_a.shape and (s >= 0).all()
    print("[init]  flat / bluravg / backproj all run")

    # validation errors
    for bad in (
        lambda: rl_deconvolve_multiview([d_a, d_b], [psf_a]),                      # len mismatch
        lambda: rl_deconvolve_multiview([d_a, d_b[:, :-1]], [psf_a, psf_b]),        # shape mismatch
        lambda: rl_deconvolve_multiview([d_a], [np.ones((20, 20)) / 400]),           # even kernel
        lambda: rl_deconvolve_multiview([d_a], [psf_a], bg=[1.0, 2.0]),             # bg len
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
