#!/usr/bin/env python3
"""Self-test for calmutils.restoration.rl_deconvolve_multiview_grid (views
sampled on different grids).

Synthetic 2D (z, x) bead experiment with known ground truth in physical
units (nm), simulated by point-sampling the continuous blurred object at each
view's sample positions:

  view A: poor z / good x resolution, coarse z sampling (120 nm), 20 nm x
  view B: good z / poorer x resolution, 40 nm z, coarse x sampling (40 nm),
          offset by a non-integer number of object voxels on both axes

Checks: kernel resampling, reduction to the core on a common grid, bead
positions in physical units (catches sign errors of the sub-voxel offsets),
best-of-both resolution vs. the same experiment on a common fine grid,
coverage, extents, and input validation. Run directly:

    $ python test_restoration_multiview_grid.py

Prints per-section results and "TEST PASSED" / "TEST FAILED".
"""
import numpy as np

from calmutils.restoration import (rl_deconvolve_multiview,
                                   rl_deconvolve_multiview_grid, object_grid,
                                   kernel_on_object_grid)

BG = 2.0
N_ITER = 100

SIG_A = np.array([300.0, 25.0])   # nm, (z, x)
SIG_B = np.array([80.0, 60.0])
SP_A, OR_A, SH_A = np.array([120.0, 20.0]), np.array([0.0, 0.0]), (32, 128)
SP_B, OR_B, SH_B = np.array([40.0, 40.0]), np.array([55.0, -7.0]), (94, 64)
OBJ_SPACING = np.array([40.0, 20.0])


def _beads(rng):
    zs = np.linspace(700, 3100, 4)
    xs = np.linspace(300, 2240, 6)
    p = np.array([(z, x) for z in zs for x in xs], dtype=float)
    return p + rng.uniform(-15, 15, p.shape)


def _acquire(rng, beads, sigma, spacing, origin, shape, peak):
    """Point samples of sum_j peak * gauss(x - p_j) + bg, Poisson noise."""
    z = origin[0] + np.arange(shape[0]) * spacing[0]
    x = origin[1] + np.arange(shape[1]) * spacing[1]
    clean = np.zeros(shape)
    for pz, px in beads:
        clean += peak * np.exp(-(z[:, None] - pz) ** 2 / (2 * sigma[0] ** 2)
                               - (x[None, :] - px) ** 2 / (2 * sigma[1] ** 2))
    return rng.poisson(clean + BG).astype(float)


def _kernel(sigma, spacing, nsig=4.0):
    """Gaussian sampled on a view grid, odd, centred, unit sum."""
    h = [int(np.ceil(nsig * s / d)) for s, d in zip(sigma, spacing)]
    z = np.arange(-h[0], h[0] + 1) * spacing[0]
    x = np.arange(-h[1], h[1] + 1) * spacing[1]
    k = np.exp(-z[:, None] ** 2 / (2 * sigma[0] ** 2)
               - x[None, :] ** 2 / (2 * sigma[1] ** 2))
    return k / k.sum()


def _bead_stats(u, origin, spacing, beads, half=(6, 6)):
    """Per-bead centroid error (nm) and second-moment sigma (nm) in u."""
    errs, sigs = [], []
    for p in beads:
        c = np.rint((p - origin) / spacing).astype(int)
        sl = tuple(slice(ci - h, ci + h + 1) for ci, h in zip(c, half))
        w = u[sl]
        idx = np.mgrid[sl].astype(float)
        pos = [origin[a] + idx[a] * spacing[a] for a in range(2)]
        m = w.sum()
        cen = np.array([(w * pos[a]).sum() / m for a in range(2)])
        sig = np.array([np.sqrt((w * (pos[a] - cen[a]) ** 2).sum() / m)
                        for a in range(2)])
        errs.append(cen - p)
        sigs.append(sig)
    return np.array(errs), np.array(sigs)


def main():
    ok = True
    rng = np.random.default_rng(7)

    # --- kernel resampling: width scales, centroid at -frac --------------
    k0 = _kernel([2.0, 1.5], [1.0, 1.0])
    for r, f in (((2, 1), (0.0, 0.0)), ((3, 2), (0.3, -0.4))):
        k = kernel_on_object_grid(k0, r, (1, 1), f)
        zz, xx = np.mgrid[:k.shape[0], :k.shape[1]]
        c = np.array([(k * zz).sum(), (k * xx).sum()]) - np.array(k.shape) // 2
        s = np.sqrt([(k * (zz - k.shape[0] // 2 - c[0]) ** 2).sum(),
                     (k * (xx - k.shape[1] // 2 - c[1]) ** 2).sum()])
        good = np.allclose(c, -np.array(f), atol=1e-3) and \
            np.allclose(s, [2.0 * r[0], 1.5 * r[1]], rtol=1e-2)
        print(f"[kernel] ratio {r} frac {f}: centroid {np.round(c, 4)}, "
              f"sigma {np.round(s, 3)} -> {'ok' if good else 'WRONG'}")
        ok &= good

    beads = _beads(rng)
    d_a = _acquire(rng, beads, SIG_A, SP_A, OR_A, SH_A, 150)
    d_b = _acquire(rng, beads, SIG_B, SP_B, OR_B, SH_B, 150)
    k_a, k_b = _kernel(SIG_A, SP_A), _kernel(SIG_B, SP_B)

    # --- grid + placement --------------------------------------------------
    grid = object_grid([SH_A, SH_B], [SP_A, SP_B], [OR_A, OR_B])
    print(f"[grid]  union: origin {grid.origin}, spacing {grid.spacing}, "
          f"shape {grid.shape}")
    ok &= np.allclose(grid.spacing, OBJ_SPACING) and np.allclose(grid.origin, [0, -20])
    ok &= grid.shape == (96, 129)
    gi = object_grid([SH_A, SH_B], [SP_A, SP_B], [OR_A, OR_B], extent="intersection")
    print(f"[grid]  intersection: origin {gi.origin}, shape {gi.shape}")
    ok &= np.allclose(gi.origin, [80, 0]) and gi.shape == (92, 126)

    # --- reduction: common grid == core with the same init ---------------
    sp = OBJ_SPACING
    c_a = _acquire(rng, beads, SIG_A, sp, OR_A, (96, 128), 150)
    c_b = _acquire(rng, beads, SIG_B, sp, OR_A, (96, 128), 150)
    ck_a, ck_b = _kernel(SIG_A, sp), _kernel(SIG_B, sp)
    init = np.full(c_a.shape, 1.0)
    r_grid, _ = rl_deconvolve_multiview_grid([c_a, c_b], [ck_a, ck_b], [sp, sp],
                                             [OR_A, OR_A], bg=BG, num_iter=20,
                                             init=init)
    r_core = rl_deconvolve_multiview([c_a, c_b], [ck_a, ck_b], bg=BG,
                                     num_iter=20, init=init)
    same = np.allclose(r_grid, r_core, rtol=1e-6, atol=1e-9)
    print(f"[reduce] common grid matches core: {same}")
    ok &= same

    # --- the real case: different grids ------------------------------------
    u, g, info = rl_deconvolve_multiview_grid(
        [d_a, d_b], [k_a, k_b], [SP_A, SP_B], [OR_A, OR_B], bg=BG,
        num_iter=N_ITER, return_info=True)
    u_a, g_a = rl_deconvolve_multiview_grid([d_a], [k_a], [SP_A], [OR_A], bg=BG,
                                            num_iter=N_ITER, spacing=OBJ_SPACING)
    u_b, g_b = rl_deconvolve_multiview_grid([d_b], [k_b], [SP_B], [OR_B], bg=BG,
                                            num_iter=N_ITER, spacing=OBJ_SPACING)
    u_c = rl_deconvolve_multiview([c_a, c_b], [ck_a, ck_b], bg=BG, num_iter=N_ITER)

    print(f"[place] steps {[p.steps for p in info.placements]}, "
          f"frac {[np.round(p.frac, 3).tolist() for p in info.placements]}, "
          f"scales {np.round(info.rl.view_scales, 3)}")
    ok &= info.placements[0].steps == (3, 1) and info.placements[1].steps == (1, 2)
    ok &= np.allclose(info.placements[0].frac, 0)
    # B: z (55 - 0) / 40 = 1.375 -> m 1, f +0.375; x (-7 - -20) / 20 = 0.65
    # -> m 1, f -0.35 (the union grid starts at x = -20)
    ok &= np.allclose(info.placements[1].frac, [0.375, -0.35])

    res = {}
    for name, uu, gg in (("A-only", u_a, g_a), ("B-only", u_b, g_b),
                         ("joint", u, g), ("common", u_c, None)):
        origin = OR_A if gg is None else gg.origin
        e, s = _bead_stats(uu, origin, OBJ_SPACING, beads)
        res[name] = (e, s)
        print(f"[beads] {name:7s} sigma z={s[:, 0].mean():6.1f} x={s[:, 1].mean():5.1f} nm | "
              f"centroid err mean z={e[:, 0].mean():+5.1f} x={e[:, 1].mean():+5.1f}, "
              f"rms z={np.sqrt((e[:, 0] ** 2).mean()):4.1f} "
              f"x={np.sqrt((e[:, 1] ** 2).mean()):4.1f} nm")

    # positions in physical units: no bias (a sign error of the sub-voxel
    # offset of B would show as 2*frac*spacing = 30 nm z / 14 nm x)
    for name in ("B-only", "joint", "A-only"):
        e = res[name][0]
        ok &= abs(e[:, 0].mean()) < 6.0 and abs(e[:, 1].mean()) < 3.0
    # best of both: z like B, x like A, clearly better than the worse view.
    # Loose vs. the single views: at equal iterations joint RL lags the best
    # view along its good axis (here ~2x the iterations in z: B's correction
    # is averaged with A's nearly flat one) - a property of multiview RL,
    # not of the grid handling, so the tight check is against the common grid
    sj, sa, sb, sc = (res[n][1].mean(0) for n in ("joint", "A-only", "B-only", "common"))
    ok &= sj[0] < sb[0] * 1.35 and sj[0] < 0.4 * sa[0]
    ok &= sj[1] < sa[1] * 1.2 and sj[1] < 0.6 * sb[1]
    # close to the same experiment sampled on a common fine grid
    ok &= np.all(np.abs(sj / sc - 1) < 0.15)

    # flux: in units of an average view, comparable to the common-grid run
    print(f"[flux]  joint sum / common-grid sum = {u.sum() / u_c.sum():.3f}")
    ok &= 0.8 < u.sum() / u_c.sum() < 1.25

    # --- coverage ------------------------------------------------------------
    cov = info.coverage
    inner = (slice(25, 70), slice(20, 110))
    print(f"[cover] interior mean {np.round(cov[(slice(None),) + inner].mean((1, 2)), 3)}, "
          f"std {np.round(cov[(slice(None),) + inner].std((1, 2)), 4)}")
    ok &= cov.shape == (2,) + g.shape
    ok &= np.allclose(cov[(slice(None),) + inner].mean((1, 2)), 1.0, atol=0.02)
    ok &= np.all(cov[(slice(None),) + inner].std((1, 2)) < 0.02)

    # --- partial FOV + weights/NaN: B covers only the left half of x --------
    d_b_half = d_b[:, :32].copy()
    w_b = np.ones(d_b_half.shape)
    w_b[:3] = 0.0
    d_b_half[:3] = np.nan
    u_h, g_h, inf_h = rl_deconvolve_multiview_grid(
        [d_a, d_b_half], [k_a, k_b], [SP_A, SP_B], [OR_A, OR_B], bg=BG,
        weights=[None, w_b], num_iter=N_ITER, return_info=True)
    n_views = (inf_h.coverage > 0.5).sum(0)
    left = beads[:, 1] < 1000
    right = beads[:, 1] > 1500
    _, s_h = _bead_stats(u_h, g_h.origin, OBJ_SPACING, beads)
    print(f"[half]  views per voxel left/right: {n_views[48, 20]}/{n_views[48, 110]}; "
          f"sigma z left {s_h[left, 0].mean():.1f} vs right {s_h[right, 0].mean():.1f} nm "
          f"(A-only {sa[0]:.1f})")
    ok &= bool(np.isfinite(u_h).all()) and g_h.shape == g.shape
    ok &= n_views[48, 20] == 2 and n_views[48, 110] == 1
    ok &= s_h[left, 0].mean() < 1.2 * sj[0] and s_h[right, 0].mean() > 0.8 * sa[0]

    # --- intersection extent runs, shape matches the grid ---------------------
    u_i, g_i = rl_deconvolve_multiview_grid([d_a, d_b], [k_a, k_b], [SP_A, SP_B],
                                            [OR_A, OR_B], bg=BG, num_iter=5,
                                            extent="intersection", padding="extend")
    ok &= u_i.shape == g_i.shape == (92, 126)

    # --- validation -----------------------------------------------------------
    for bad in (
        lambda: rl_deconvolve_multiview_grid([d_a, d_b], [k_a, k_b], [SP_A, [40.0, 30.0]],
                                             [OR_A, OR_B]),                 # non-integer step
        lambda: rl_deconvolve_multiview_grid([d_a, d_b], [k_a, k_b[:-1]], [SP_A, SP_B],
                                             [OR_A, OR_B]),                 # even kernel
        lambda: rl_deconvolve_multiview_grid([d_a, d_b], [k_a, k_b], [SP_A, SP_B],
                                             [OR_A, OR_B + 1e5]),           # no shared footprint
        lambda: rl_deconvolve_multiview_grid([d_a, d_b], [k_a, k_b], [SP_A, SP_B],
                                             [OR_A, OR_B], weights=np.ones(SH_A)),  # one weight array
        lambda: rl_deconvolve_multiview_grid([d_a, d_b], [k_a, k_b], [SP_A],
                                             [OR_A, OR_B]),                 # spacings len
        lambda: rl_deconvolve_multiview_grid([d_a, d_b], [k_a, k_b], [SP_A, SP_B],
                                             [OR_A, OR_B], extent=([0, 0], [-1, 0])),  # bad box
        lambda: rl_deconvolve_multiview_grid([d_a, d_b], [k_a, k_b], [SP_A, SP_B],
                                             [OR_A, OR_B], init=np.ones(SH_A)),  # init shape
    ):
        try:
            bad()
            ok = False
            print("[validate] MISSING expected error")
        except ValueError:
            pass
    print("[validate] input validation raises as expected")

    print("TEST PASSED" if ok else "TEST FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
