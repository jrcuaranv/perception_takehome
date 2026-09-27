"""Does the k=3 conclusion depend on how much the camera rotates? Stratify by per-frame angular speed and pair rotation."""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from perception.geometry import project, transform, unproject
from perception.scene import Scene

sc = Scene(ROOT / "data/scene_a")
cam = sc.depth_cam
LO, HI, GAP = 135, 366, 8


def rot_deg(a, b):
    R = sc.pose(a)[:3, :3].T @ sc.pose(b)[:3, :3]
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def reproj(i, j, k):
    di, dj = sc.depth_m(i + k), sc.depth_m(j + k)
    pts, _ = unproject(di, cam, stride=2)
    uv, z = project(transform(np.linalg.inv(sc.pose(j)) @ sc.pose(i), pts), cam)
    u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
    ok = (u >= 0) & (u < 256) & (v >= 0) & (v < 192) & (z > 0.2)
    zm = dj[v[ok], u[ok]]
    g = zm > 0.2
    r = np.abs(z[ok][g] - zm[g])
    r = r[r < 0.5]
    return (np.median(r), int(ok.sum())) if len(r) > 50 else (np.nan, int(ok.sum()))


def edge_corr(i, k):
    g = cv2.cvtColor(sc.rgb(i), cv2.COLOR_RGB2GRAY).astype(np.float32)
    d = sc.depth_m(i + k)
    gm = cv2.GaussianBlur(np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)), (0, 0), 1)
    dm = np.hypot(cv2.Sobel(d, cv2.CV_32F, 1, 0), cv2.Sobel(d, cv2.CV_32F, 0, 1))
    return np.corrcoef(gm.ravel(), np.log1p(dm.ravel() * 10))[0, 1]


frames = np.arange(LO, HI + 1)
ang = np.array([rot_deg(i, i + 1) for i in frames]) * 10  # deg/s
print(f"block angular speed (deg/s): median {np.median(ang):.1f}, p90 {np.percentile(ang, 90):.1f}, max {ang.max():.1f}")
ctrl = np.arange(400, 1200)
angc = np.array([rot_deg(i, i + 1) for i in ctrl]) * 10
print(f"control angular speed (deg/s): median {np.median(angc):.1f}, p90 {np.percentile(angc, 90):.1f}, max {angc.max():.1f}")

pairs = [(int(i), int(i) + GAP) for i in frames if i + GAP <= HI]
pr = np.array([rot_deg(i, j) for i, j in pairs])
bins = [("low rotation  (<8 deg)", pr < 8), ("mid rotation  (8-20 deg)", (pr >= 8) & (pr < 20)), ("high rotation (>=20 deg)", pr >= 20)]
print("\nreprojection residual (m) by rotation between the two frames of the pair; median over pairs")
print(f"{'stratum':26s} {'n':>4s} {'ovlp px':>8s} " + " ".join(f"k={k:+d}".rjust(8) for k in range(0, 6)))
for name, m in bins:
    idx = np.where(m)[0]
    if len(idx) < 5:
        print(f"{name:26s} {len(idx):4d}  (too few)")
        continue
    sel = [pairs[t] for t in idx[:: max(1, len(idx) // 25)]]
    out = {k: [reproj(i, j, k) for i, j in sel] for k in range(0, 6)}
    ov = np.median([o[1] for o in out[3]])
    print(f"{name:26s} {len(idx):4d} {ov:8.0f} " + " ".join(f"{np.nanmedian([o[0] for o in out[k]]):8.4f}" for k in range(0, 6)))

print("\nedge correlation (no poses) by per-frame angular speed")
print(f"{'stratum':26s} {'n':>4s} " + " ".join(f"k={k:+d}".rjust(8) for k in range(0, 6)))
q = np.percentile(ang, [50])
for name, m in (("slow (<= median speed)", ang <= q[0]), ("fast (> median speed)", ang > q[0])):
    sel = frames[m][:: max(1, m.sum() // 40)]
    print(f"{name:26s} {len(sel):4d} " + " ".join(f"{np.mean([edge_corr(int(i), k) for i in sel]):8.4f}" for k in range(0, 6)))
