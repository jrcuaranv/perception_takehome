"""Before/after: reprojection residual and RGB/depth edge correlation, scene_a vs scene_a_fixed, on the same frames."""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from perception.geometry import project, transform, unproject
from perception.scene import Scene


def reproj(sc, i, j):
    cam = sc.depth_cam
    di, dj = sc.depth_m(i), sc.depth_m(j)
    pts, _ = unproject(di, cam, stride=2)
    if len(pts) < 200:
        return np.nan
    uv, z = project(transform(np.linalg.inv(sc.pose(j)) @ sc.pose(i), pts), cam)
    u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
    ok = (u >= 0) & (u < 256) & (v >= 0) & (v < 192) & (z > 0.2)
    zm = dj[v[ok], u[ok]]
    g = zm > 0.2
    r = np.abs(z[ok][g] - zm[g])
    r = r[r < 0.5]
    return np.median(r) if len(r) > 50 else np.nan


def edge_corr(sc, i):
    g = cv2.cvtColor(sc.rgb(i), cv2.COLOR_RGB2GRAY).astype(np.float32)
    d = sc.depth_m(i)
    gm = cv2.GaussianBlur(np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)), (0, 0), 1)
    dm = np.hypot(cv2.Sobel(d, cv2.CV_32F, 1, 0), cv2.Sobel(d, cv2.CV_32F, 0, 1))
    return np.corrcoef(gm.ravel(), np.log1p(dm.ravel() * 10))[0, 1]


orig, fixed = Scene(ROOT / "data/scene_a"), Scene(ROOT / "data/scene_a_fixed")
rng = np.random.default_rng(7)  # different seed from the sweep that chose k
sets = {
    "block 135-369 (fixed)": np.arange(135, 362),
    "clean 400-1200 (control, unchanged)": np.arange(400, 1200),
}
print(f"{'frames':38s} {'metric':22s} {'scene_a':>9s} {'scene_a_fixed':>14s}")
for label, fr in sets.items():
    pr = [(int(i), int(i) + 8) for i in rng.choice(fr, 60, replace=False)]
    ef = [int(i) for i in rng.choice(fr, 80, replace=False)]
    for m, f in (("reproj residual (m)", lambda s: np.nanmedian([reproj(s, i, j) for i, j in pr])),
                 ("edge correlation", lambda s: np.mean([edge_corr(s, i) for i in ef]))):
        print(f"{label:38s} {m:22s} {f(orig):9.4f} {f(fixed):14.4f}")
