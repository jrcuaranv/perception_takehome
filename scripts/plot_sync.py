"""Plot depth_stamp - colour_stamp (seconds) per frame, one panel per scene."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
scenes = [s["name"] for s in json.loads((ROOT / "variant/scenes.json").read_text())["scenes"]]

INK, MUTED, GRID, MARK = "#1f2933", "#6b7785", "#e4e8ec", "#2a6fbb"
fig, axes = plt.subplots(len(scenes), 1, figsize=(10, 2.3 * len(scenes)), sharey=True)
for ax, name in zip(axes, scenes):
    pairs = np.array(json.loads((ROOT / f"variant/{name}/sync.json").read_text())["pairs"], dtype=float)
    off = pairs[:, 1] - pairs[:, 0]
    i = np.arange(len(off))
    ax.axhline(0, color=MUTED, lw=1)
    ax.plot(i, off, color=MARK, lw=1, marker="o", ms=2.5, mfc=MARK, mec="none")
    ax.set_yscale("symlog", linthresh=0.05)
    ax.set_yticks([0, -0.1, -0.3, -1, -5])
    ax.set_yticklabels(["0", "-0.1", "-0.3", "-1", "-5"])
    ax.set_ylim(-8, 0.03)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_title(f"{name}: {(off < -0.05).sum()} of {len(off)} frames paired with an earlier depth frame",
                 loc="left", fontsize=10, color=INK)
    ax.tick_params(colors=MUTED, labelsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
axes[-1].set_xlabel("frame index", color=MUTED, fontsize=9)
fig.supylabel("depth stamp - colour stamp (s, symlog)", color=MUTED, fontsize=9)
fig.tight_layout()
out = ROOT / "deliverables/plots/sync_offset.png"
fig.savefig(out, dpi=140)
print(out)
