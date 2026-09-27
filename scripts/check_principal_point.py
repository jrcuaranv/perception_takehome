"""RGB/depth edge alignment vs a 2D shift of the depth image (pose-free).
With ~identity T_color_depth, a colour principal point differing from the depth one by (dcx, dcy) predicts
that depth pixel (u, v) lands on colour pixel (u + dcx, v + dcy). We sweep the shift and compare the peak to that prediction.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from perception.scene import Scene

R = 26  # sweep -R..R px
# frames known to be paired at offset 0 (scene_a block and scene_c frozen block excluded)
RANGES = {"scene_a": (400, 1200), "scene_b": (0, 1080), "scene_c": (110, 750), "scene_d": (0, 1160)}
rng = np.random.default_rng(0)


def grads(sc, i):
    g = cv2.cvtColor(sc.rgb(i), cv2.COLOR_RGB2GRAY).astype(np.float32)
    d = sc.depth_m(i)
    gm = cv2.GaussianBlur(np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)), (0, 0), 1)
    dm = np.log1p(10 * np.hypot(cv2.Sobel(d, cv2.CV_32F, 1, 0), cv2.Sobel(d, cv2.CV_32F, 0, 1)))
    return gm, dm


def corr_at(gm, dm, sx, sy):
    """Correlation when depth pixel (u,v) is placed at colour pixel (u+sx, v+sy), over the overlap only."""
    H, W = gm.shape
    x0, x1 = max(0, sx), min(W, W + sx)
    y0, y1 = max(0, sy), min(H, H + sy)
    a = gm[y0:y1, x0:x1]
    b = dm[y0 - sy : y1 - sy, x0 - sx : x1 - sx]
    return np.corrcoef(a.ravel(), b.ravel())[0, 1]


for s in ("scene_a", "scene_b", "scene_c", "scene_d"):
    sc = Scene(ROOT / "data" / s)
    lo, hi = RANGES[s]
    fr = rng.choice(np.arange(lo, hi), 30, replace=False)
    G = [grads(sc, int(i)) for i in fr]
    S = np.zeros((2 * R + 1, 2 * R + 1))
    for iy, sy in enumerate(range(-R, R + 1)):
        for ix, sx in enumerate(range(-R, R + 1)):
            S[iy, ix] = np.mean([corr_at(g, d, sx, sy) for g, d in G])
    iy, ix = np.unravel_index(S.argmax(), S.shape)
    px, py = ix - R, iy - R
    dcx, dcy = sc.color["cx"] - sc.depth_cam["cx"], sc.color["cy"] - sc.depth_cam["cy"]
    pred = (int(round(dcx)), int(round(dcy)))
    at = lambda x, y: S[y + R, x + R]
    print(f"{s}: peak shift (dx,dy)=({px},{py}) corr {S.max():.3f} | predicted by calib ({dcx:+.1f},{dcy:+.1f}) corr there {at(*pred):.3f} | at (0,0) {at(0,0):.3f}")
