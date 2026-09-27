"""Show RGB, its blurred gradient map, depth, and its log-compressed gradient map for a few sample frames per
scene -- the inputs to the RGB/depth edge-correlation checks used throughout Part 1."""
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from perception.scene import Scene

# frames outside the known mis-paired blocks, so what's shown reflects an ordinary well-paired frame
SAMPLES = {
    "scene_a": [420, 650, 900, 1150],
    "scene_b": [100, 350, 600, 900],
    "scene_c": [200, 350, 500, 700],
    "scene_d": [100, 400, 700, 1000],
}


def grads(rgb, depth_m):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gm = cv2.GaussianBlur(np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)), (0, 0), 1)
    dm = np.log1p(10 * np.hypot(cv2.Sobel(depth_m, cv2.CV_32F, 1, 0), cv2.Sobel(depth_m, cv2.CV_32F, 0, 1)))
    return gm, dm


for name, frames in SAMPLES.items():
    sc = Scene(ROOT / "data" / name)
    fig, axes = plt.subplots(len(frames), 4, figsize=(11, 2.6 * len(frames)))
    for row, i in enumerate(frames):
        rgb, depth = sc.rgb(i), sc.depth_m(i)
        gm, dm = grads(rgb, depth)
        corr = np.corrcoef(gm.ravel(), dm.ravel())[0, 1]
        panels = [
            (rgb, None, None, "RGB"),
            (gm, "inferno", None, "RGB edge map (blurred |grad|)"),
            (np.ma.masked_where(depth <= 0, depth), "viridis", 5.0, "depth (m)"),
            (dm, "inferno", None, "depth edge map (log |grad|)"),
        ]
        for col, (im, cmap, vmax, title) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(im, cmap=cmap, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(title, fontsize=9)
        axes[row, 0].set_ylabel(f"frame {i}\ncorr={corr:.3f}", fontsize=8, rotation=0, ha="right", va="center", labelpad=35)
    fig.suptitle(f"{name}: RGB/depth edge maps behind the edge-correlation check", fontsize=12)
    fig.tight_layout(rect=[0.02, 0, 1, 0.96])
    out = ROOT / f"deliverables/plots/edge_maps_{name}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(out)
