"""How plausible are query_targets.json's 'visible_targets'? For every (query, frame, target) labelled visible on the dev
scenes, project the surveyed box into the frame (pose + the depth intrinsics our lift uses) and measure:
  in_view : fraction of the box's projected 2D extent that falls inside the image (1 = fully in view)
  front   : measured depth at the projected centre minus expected depth (m); < 0 means something is in front of it
Geometry-dependent (uses the poses, so scene_a's suspected shift can move these), so read as a plausibility check."""
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from perception.geometry import project, transform
from perception.scene import Scene

rows = []
for name in ("scene_a", "scene_b"):
    sc = Scene(ROOT / "data" / name)
    boxes = {b["uid"]: b for b in sc.boxes}
    H, W = sc.color["height"], sc.color["width"]
    tg = json.loads((ROOT / "variant" / name / "query_targets.json").read_text())
    seen = set()
    for qid, t in tg.items():
        for f, fr in t["frames"].items():
            for uid in fr["visible_targets"]:
                if (f, uid) in seen:
                    continue
                seen.add((f, uid))
                b = boxes[uid]
                c, size, axes = np.array(b["center"]), np.array(b["size"]), np.array(b["axes"])
                corners = np.array([c + sum(s * size[i] / 2 * axes[i] for i, s in enumerate(sg)) for sg in product([-1, 1], repeat=3)])
                P = sc.pose(int(f))
                cam = transform(np.linalg.inv(P), np.vstack([corners, c[None]]))
                uv, z = project(cam, sc.depth_cam)   # same intrinsics our lift uses (pixel-aligned depth)
                if (z[:8] <= 0.05).all():
                    rows.append((name, qid, f, uid, b["label"], 0.0, np.nan)); continue
                ok = z[:8] > 0.05                       # corners behind the camera are ignored (approximation)
                x0, x1, y0, y1 = uv[:8][ok, 0].min(), uv[:8][ok, 0].max(), uv[:8][ok, 1].min(), uv[:8][ok, 1].max()
                area = max((x1 - x0) * (y1 - y0), 1e-6)
                inter = max(0, min(x1, W) - max(x0, 0)) * max(0, min(y1, H) - max(y0, 0))
                front = np.nan
                u, v = int(round(uv[8, 0])), int(round(uv[8, 1]))
                if z[8] > 0 and 0 <= u < W and 0 <= v < H:
                    d = sc.depth_m(int(f))[v, u]
                    front = d - z[8] if d > 0 else np.nan
                rows.append((name, qid, f, uid, b["label"], inter / area, front))

print(f"{'scene':8s} {'pairs':>5s} {'>=80% in view':>14s} {'<50% in view':>13s} {'<20% in view':>13s} {'centre occluded (>0.5 m in front)':>34s}")
for name in ("scene_a", "scene_b"):
    r = [x for x in rows if x[0] == name]
    iv = np.array([x[5] for x in r]); fr = np.array([x[6] for x in r], float)
    print(f"{name:8s} {len(r):5d} {int((iv>=0.8).sum()):14d} {int((iv<0.5).sum()):13d} {int((iv<0.2).sum()):13d} {int((fr<-0.5).sum()):>34d}")
print("\nlabelled visible but <50% of the box inside the image:")
for x in sorted(rows):
    if x[5] < 0.5:
        print(f"  {x[0]} {x[1]} f={x[2]:>4s} {x[4]:12s} {x[3]}  in_view={x[5]:.2f}")
