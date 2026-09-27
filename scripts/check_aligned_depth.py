"""Does warping depth into the colour camera with the recorded calibration (depth intrinsics + T_color_depth + colour
intrinsics) improve RGB/depth edge alignment over treating depth as pixel-aligned with colour? Pose-free.
Correlation is computed only on pixels where the warped depth is valid (eroded 5x5, so holes/borders are ignored),
and the raw-depth number uses the same mask, so both columns are compared on identical pixels."""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from perception.geometry import aligned_depth_to_color
from perception.scene import Scene

# frames paired at offset 0 (scene_a lag block and scene_c frozen block excluded)
RANGES = {"scene_a": (400, 1200), "scene_b": (0, 1080), "scene_c": (110, 750), "scene_d": (0, 1160)}
N_FRAMES = 30


def edge_corr(rgb, d, mask):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gm = cv2.GaussianBlur(np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)), (0, 0), 1)
    dm = np.log1p(10 * np.hypot(cv2.Sobel(d, cv2.CV_32F, 1, 0), cv2.Sobel(d, cv2.CV_32F, 0, 1)))
    return np.corrcoef(gm[mask], dm[mask])[0, 1]


rng = np.random.default_rng(0)
print(f"{'scene':8s} {'raw (pixel-aligned)':>20s} {'warped (recorded calib)':>24s} {'valid px':>9s}")
for name, (lo, hi) in RANGES.items():
    sc = Scene(ROOT / "data" / name)
    raw_c, warp_c, frac = [], [], []
    for i in rng.choice(np.arange(lo, hi), N_FRAMES, replace=False):
        i = int(i)
        warped = aligned_depth_to_color(sc.depth_raw(i), sc.depth_cam, sc.color, sc.T_color_depth)
        mask = cv2.erode((warped > 0).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        rgb = sc.rgb(i)
        frac.append(mask.mean())
        raw_c.append(edge_corr(rgb, sc.depth_m(i), mask))
        warp_c.append(edge_corr(rgb, warped, mask))
    print(f"{name:8s} {np.mean(raw_c):20.3f} {np.mean(warp_c):24.3f} {np.mean(frac):9.2f}")
