"""Plot each scene's camera trajectory in 3D: path, a camera-axis triad every N frames, and the world frame origin."""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from perception.scene import Scene

INK, MUTED, PATH = "#1f2933", "#6b7785", "#9aa5b1"
AX_COLOR = {"x": "#c0392b", "y": "#1e8a4c", "z": "#2a6fbb"}  # standard R/G/B for X/Y/Z
N_TRIADS = 16

scenes = [s["name"] for s in json.loads((ROOT / "variant/scenes.json").read_text())["scenes"]]
fig = plt.figure(figsize=(13, 12))

for k, name in enumerate(scenes):
    sc = Scene(ROOT / "data" / name)
    poses = np.array([sc.pose(i) for i in range(len(sc))])
    pos = poses[:, :3, 3]

    ax = fig.add_subplot(2, 2, k + 1, projection="3d")
    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color=PATH, lw=1.2, zorder=1)
    ax.scatter(*pos[0], color="#1e8a4c", s=35, label="start", zorder=3)
    ax.scatter(*pos[-1], color="#c0392b", s=35, label="end", zorder=3)

    # scale for the little axis triads: a fraction of the trajectory's own extent
    extent = (pos.max(0) - pos.min(0))
    L = 0.08 * np.linalg.norm(extent)
    idx = np.linspace(0, len(sc) - 1, N_TRIADS).round().astype(int)
    for i in idx:
        o = poses[i, :3, 3]
        for a, axis in enumerate("xyz"):
            d = poses[i, :3, a]  # camera axis direction in world frame
            ax.plot(*np.stack([o, o + L * d], axis=1), color=AX_COLOR[axis], lw=1.6, zorder=2)

    # world frame origin, drawn larger so it reads as "the" frame, not just another triad
    Lw = 0.18 * np.linalg.norm(extent)
    origin = np.zeros(3)
    if np.linalg.norm(pos - origin, axis=1).min() < 2 * Lw:
        origin = pos.min(0) - 0.15 * extent  # move the glyph clear of the path if the path passes near (0,0,0)
    for a, axis in enumerate("xyz"):
        d = np.eye(3)[a]
        ax.plot(*np.stack([origin, origin + Lw * d], axis=1), color=AX_COLOR[axis], lw=3.2, zorder=4)
    ax.scatter(*origin, color=INK, s=25, marker="o", zorder=5)
    ax.text(*(origin + 1.15 * Lw * np.eye(3)[0]), "x", color=AX_COLOR["x"], fontsize=9, zorder=6)
    ax.text(*(origin + 1.15 * Lw * np.eye(3)[1]), "y", color=AX_COLOR["y"], fontsize=9, zorder=6)
    ax.text(*(origin + 1.15 * Lw * np.eye(3)[2]), "z", color=AX_COLOR["z"], fontsize=9, zorder=6)

    allp = np.vstack([pos, origin[None]])
    ctr, rng = (allp.max(0) + allp.min(0)) / 2, (allp.max(0) - allp.min(0)).max() / 2 * 1.15
    ax.set_xlim(ctr[0] - rng, ctr[0] + rng)
    ax.set_ylim(ctr[1] - rng, ctr[1] + rng)
    ax.set_zlim(ctr[2] - rng, ctr[2] + rng)
    ax.set_box_aspect([1, 1, 1])

    ax.set_title(f"{name}  ({len(sc)} frames)", color=INK, fontsize=11, loc="left")
    ax.set_xlabel("x (m)", color=MUTED, fontsize=8)
    ax.set_ylabel("y (m)", color=MUTED, fontsize=8)
    ax.set_zlabel("z (m)", color=MUTED, fontsize=8)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.xaxis.pane.set_alpha(0.03)
    ax.yaxis.pane.set_alpha(0.03)
    ax.zaxis.pane.set_alpha(0.03)
    ax.legend(loc="upper left", fontsize=7, frameon=False)

fig.suptitle("Camera trajectory per scene — coloured triads are camera axes (x red, y green, z blue),\n"
             "thick triad is the world-frame origin", fontsize=12, color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = ROOT / "deliverables/plots/trajectories.png"
fig.savefig(out, dpi=140)
print(out)
plt.show()