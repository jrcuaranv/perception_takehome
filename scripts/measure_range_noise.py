"""Measure how the multi-view depth reprojection residual grows with range, per scene, and fit a noise model to it.

Same residual as Part 1's sync/focal checks (unproject frame i's depth, move it into frame j with the poses, compare
against frame j's measured depth) but here every pixel keeps its own range z_i (from frame i) and residual, instead of
collapsing to one median per frame pair. Binning by z_i then shows residual(z) directly.

Pairs are restricted to <15 deg rotation so a range effect isn't confounded with the rotation-floor effect found in
Part 1 (scripts/check_rotation_bias.py). Uses scene_a_fixed / scene_c_fixed so the sync-fault frames (which contain
no valid depth there) simply drop out rather than contaminating the fit.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from perception.geometry import project, transform, unproject
from run_part2 import load_scene

GAP = 10
MAX_ROT_DEG = 15
N_PAIRS = 120
BINS = np.array([0.3, 0.6, 0.9, 1.2, 1.6, 2.0, 2.5, 3.0, 4.0, 6.0])


def rot_deg(sc, i, j):
    R = sc.pose(i)[:3, :3].T @ sc.pose(j)[:3, :3]
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def pair_residuals(sc, i, j):
    """Per-pixel (range_in_frame_i, |residual|) for one frame pair."""
    cam = sc.depth_cam
    di, dj = sc.depth_m(i), sc.depth_m(j)
    pts, _ = unproject(di, cam, stride=2)
    if len(pts) < 200:
        return np.empty(0), np.empty(0)
    z_i = pts[:, 2]
    T = np.linalg.inv(sc.pose(j)) @ sc.pose(i)
    uv, z_j_pred = project(transform(T, pts), cam)
    u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
    ok = (u >= 0) & (u < cam["width"]) & (v >= 0) & (v < cam["height"]) & (z_j_pred > 0.2)
    z_meas = dj[v[ok], u[ok]]
    good = z_meas > 0.2
    r = np.abs(z_j_pred[ok][good] - z_meas[good])
    z_i = z_i[ok][good]
    keep = r < 0.5  # drop occlusion outliers, same threshold as Part 1
    return z_i[keep], r[keep]


def measure(scene_name):
    sc = load_scene(scene_name)
    n = len(sc)
    rng = np.random.default_rng(0)
    idx = rng.choice(np.arange(0, n - GAP), min(N_PAIRS, n - GAP), replace=False)
    all_z, all_r = [], []
    for i in idx:
        i = int(i)
        if rot_deg(sc, i, i + GAP) > MAX_ROT_DEG:
            continue
        z, r = pair_residuals(sc, i, i + GAP)
        all_z.append(z)
        all_r.append(r)
    z, r = np.concatenate(all_z), np.concatenate(all_r)

    # binned medians (robust to the same per-pixel outliers the rest of Part 1 guarded against)
    which = np.digitize(z, BINS)
    med_z, med_r, n_pix = [], [], []
    for b in range(len(BINS) - 1):
        m = which == b + 1
        if m.sum() < 200:
            continue
        med_z.append(np.median(z[m]))
        med_r.append(np.median(r[m]))
        n_pix.append(int(m.sum()))
    med_z, med_r, n_pix = np.array(med_z), np.array(med_r), np.array(n_pix)

    # fit residual = k*z (through the origin) and residual = a + k*z, both weighted by pixel count per bin
    k_through_origin = np.average(med_r / med_z, weights=n_pix)
    A = np.stack([med_z, np.ones_like(med_z)], axis=1)
    (k_lin, a_lin), *_ = np.linalg.lstsq(A * np.sqrt(n_pix)[:, None], med_r * np.sqrt(n_pix), rcond=None)
    # how much of the range-dependence is real vs a flat floor: compare a constant-only fit
    const = np.average(med_r, weights=n_pix)
    rmse = lambda pred: np.sqrt(np.average((med_r - pred) ** 2, weights=n_pix))
    return dict(med_z=med_z, med_r=med_r, n_pix=n_pix, k0=k_through_origin, k_lin=k_lin, a_lin=a_lin,
                rmse_const=rmse(const), rmse_lin=rmse(k_lin * med_z + a_lin), rmse_k0=rmse(k_through_origin * med_z),
                const=const, n_total=len(z))


for name in ("scene_a", "scene_b", "scene_c", "scene_d"):
    d = measure(name)
    print(f"\n{name}: {d['n_total']:,} pixels, rotation < {MAX_ROT_DEG} deg pairs only")
    print("  range bin (m): " + "  ".join(f"{z:4.2f}" for z in d["med_z"]))
    print("  residual (m):  " + "  ".join(f"{r:4.3f}" for r in d["med_r"]))
    print(f"  n pixels/bin:  " + "  ".join(f"{n:4d}" for n in d["n_pix"]))
    print(f"  fit residual = k*z (through origin):     k = {d['k0']:.4f}  (i.e. ~{d['k0']*100:.2f}% of range)   rmse={d['rmse_k0']:.4f}")
    print(f"  fit residual = a + k*z:                  a = {d['a_lin']:.4f} m, k = {d['k_lin']:.4f}            rmse={d['rmse_lin']:.4f}")
    print(f"  flat floor only (residual = const):      const = {d['const']:.4f} m                              rmse={d['rmse_const']:.4f}")
