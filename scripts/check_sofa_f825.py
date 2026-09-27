"""Is the surveyed sofa (obj_026) actually visible in scene_a frame 825? Project its box into the frame and compare with depth."""
import json
import sys
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from perception.geometry import project, transform
from perception.scene import Scene

F, UID = 825, "obj_026"
sc = Scene(ROOT / "data/scene_a")
box = next(b for b in sc.boxes if b["uid"] == UID)
c, size, axes = np.array(box["center"]), np.array(box["size"]), np.array(box["axes"])
corners = np.array([c + sum(s * size[i] / 2 * axes[i] for i, s in enumerate(signs)) for signs in product([-1, 1], repeat=3)])
P = sc.pose(F)
cam_pts = transform(np.linalg.inv(P), np.vstack([corners, c[None]]))
uv, z = project(cam_pts, sc.color)
print(f"box {UID} ({box['label']}) size {np.round(size, 2)} m; camera-to-centre distance {np.linalg.norm(c - P[:3, 3]):.2f} m")
print("corner depths in camera (m):", np.round(z[:8], 2), "| centre z:", round(float(z[8]), 2), "(z<=0 means behind the camera)")
print("corner pixels (u,v):", np.round(uv[:8]).astype(int).tolist())
H, W = sc.rgb(F).shape[:2]
inside = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H) & (z > 0)
print(f"corners inside the {W}x{H} image and in front of the camera: {int(inside[:8].sum())}/8; centre inside: {bool(inside[8])}")
if inside[8]:
    d = sc.depth_m(F)[int(round(uv[8, 1])), int(round(uv[8, 0]))]
    print(f"measured depth at the projected centre: {d:.2f} m vs expected {z[8]:.2f} m")

dets = [d for d in json.loads((ROOT / "variant/detections.json").read_text())["scenes"]["scene_a"][str(F)] if d["score"] >= 0.1]
print("detections in this frame:", [(d["label"], round(d["score"], 2)) for d in sorted(dets, key=lambda d: -d["score"])])

edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].imshow(sc.rgb(F)); ax[0].set_title(f"scene_a frame {F}: RGB")
ax[1].imshow(sc.rgb(F)); ax[1].set_title(f"projected surveyed {box['label']} {UID} (behind camera: none drawn)")
for a, b in edges:
    if z[a] > 0 and z[b] > 0:
        ax[1].plot([uv[a, 0], uv[b, 0]], [uv[a, 1], uv[b, 1]], color="#e8590c", lw=1.6)
for a in ax:
    a.set_xlim(0, W); a.set_ylim(H, 0); a.axis("off")
out = ROOT / "deliverables/plots/scene_a_f825_sofa.png"
fig.tight_layout(); fig.savefig(out, dpi=130); print(out)
