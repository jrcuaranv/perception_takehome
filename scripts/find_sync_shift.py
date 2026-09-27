"""Estimate the depth-index shift k for a block: colour i should use depth row i+k.
Two independent scores per k (lower residual / higher edge correlation is better):
  1. multi-view depth reprojection residual using the (colour-time) poses
  2. correlation between RGB gradient magnitude and depth gradient magnitude
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from perception.geometry import project, transform, unproject
from perception.scene import Scene

if len(sys.argv) not in (1, 4):
    sys.exit("usage: find_sync_shift.py [SCENE FIRST_FRAME LAST_FRAME]   (default: scene_a 135 372)")
name, lo, hi = (sys.argv[1], int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) == 4 else ("scene_a", 135, 372)
K = range(-6, 7)
sc = Scene(ROOT / "data" / name)
cam = sc.depth_cam
rng = np.random.default_rng(0)


def reproj(i, j, k):
    di, dj = sc.depth_m(i + k), sc.depth_m(j + k)
    pts, _ = unproject(di, cam, stride=2)
    T = np.linalg.inv(sc.pose(j)) @ sc.pose(i)
    uv, z = project(transform(T, pts), cam)
    u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
    ok = (u >= 0) & (u < 256) & (v >= 0) & (v < 192) & (z > 0.2)
    zm = dj[v[ok], u[ok]]
    g = zm > 0.2
    r = np.abs(z[ok][g] - zm[g])
    r = r[r < 0.5]
    return np.median(r) if len(r) > 50 else np.nan


def edge_corr(i, k):
    g = cv2.cvtColor(sc.rgb(i), cv2.COLOR_RGB2GRAY).astype(np.float32)
    d = sc.depth_m(i + k)
    gm = cv2.GaussianBlur(np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)), (0, 0), 1)
    dm = np.hypot(cv2.Sobel(d, cv2.CV_32F, 1, 0), cv2.Sobel(d, cv2.CV_32F, 0, 1))
    return np.corrcoef(gm.ravel(), np.log1p(dm.ravel() * 10))[0, 1]


# keep every shifted index inside the recording; use frames that move (parallax) for the reprojection score
frames = np.arange(lo, hi + 1)
pairs = [(int(i), int(i) + 8) for i in rng.choice(frames[frames + 8 + 6 <= min(hi + 1, len(sc) - 7)], 40, replace=False)]
ef = [int(i) for i in rng.choice(frames[frames + 6 < len(sc)], 60, replace=False)]

print(f"{name} block {lo}-{hi}   (colour i uses depth row i+k)")
print("   k   reproj residual (m)   RGB/depth edge corr")
res = {}
for k in K:
    r = np.nanmedian([reproj(i, j, k) for i, j in pairs])
    e = np.mean([edge_corr(i, k) for i in ef])
    res[k] = (r, e)
    print(f"{k:4d}   {r:8.4f}              {e:7.4f}")
print("best k by reprojection:", min(res, key=lambda k: res[k][0]), "| by edge corr:", max(res, key=lambda k: res[k][1]))
