"""10x10 grids of RGB | depth pairs (5 pairs per row) sampled evenly from a mis-paired block."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from perception.scene import Scene

BLOCKS = {"scene_a": (135, 372), "scene_c": (58, 105)}
DMAX = 5.0
cmap = plt.get_cmap("viridis").copy()
cmap.set_bad("#000000")  # no return

for name, (lo, hi) in BLOCKS.items():
    sc = Scene(ROOT / "data" / name)
    idx = np.unique(np.linspace(lo, hi, 50).round().astype(int))
    fig, axes = plt.subplots(10, 10, figsize=(20, 15.6))
    for ax in axes.ravel():
        ax.axis("off")
    for k, i in enumerate(idx):
        r, c = divmod(k, 5)
        a_rgb, a_dep = axes[r, 2 * c], axes[r, 2 * c + 1]
        a_rgb.imshow(sc.rgb(i))
        d = sc.depth_m(i)
        im = a_dep.imshow(np.ma.masked_where(d <= 0, d), cmap=cmap, vmin=0, vmax=DMAX)
        a_rgb.set_title(f"frame {i}", fontsize=8, pad=2)
    fig.suptitle(f"{name}, frames {lo}-{hi}: RGB | depth (0-{DMAX:g} m, black = no return)", fontsize=13)
    fig.subplots_adjust(left=0.01, right=0.93, top=0.95, bottom=0.01, wspace=0.03, hspace=0.18)
    cax = fig.add_axes([0.945, 0.2, 0.01, 0.6])
    fig.colorbar(im, cax=cax, label="depth (m)")
    out = ROOT / f"deliverables/plots/block_grid_{name}.png"
    fig.savefig(out, dpi=80)
    plt.close(fig)
    print(out, len(idx), "frames")
