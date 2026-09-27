"""Build data/scene_a_fixed/ and variant/scene_a/sync_fixed.json without touching scene_a.

Inside the block (default 135-372) the recorder paired each colour frame with a depth frame ~3 frames old.
For each colour frame we find the depth row whose stamp equals the colour stamp; if none exists the depth is
written as all zeros (0 = no return) so frame indices, poses and detections stay aligned.
Outside the block everything is copied as recorded.
"""
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC, DST = ROOT / "data/scene_a", ROOT / "data/scene_a_fixed"
LO, HI = 135, 372

sync = json.loads((ROOT / "variant/scene_a/sync.json").read_text())
pairs = sync["pairs"]
row_of = {}
for r, (_, d) in enumerate(pairs):
    row_of.setdefault(d, []).append(r)

if DST.exists():
    raise SystemExit(f"{DST} exists; move or delete it to rebuild")
for sub in ("rgb", "depth"):
    (DST / sub).mkdir(parents=True)

fixed_pairs, source_row, zeroed = [], [], []
for i, (c, d) in enumerate(pairs):
    src = i
    if LO <= i <= HI:
        rows = row_of.get(c, [])
        src = rows[0] if len(rows) == 1 else None
    shutil.copyfile(SRC / "rgb" / f"{i:06d}.png", DST / "rgb" / f"{i:06d}.png")
    if src is None:
        raw = cv2.imread(str(SRC / "depth" / f"{i:06d}.png"), cv2.IMREAD_UNCHANGED)
        cv2.imwrite(str(DST / "depth" / f"{i:06d}.png"), np.zeros_like(raw))
        fixed_pairs.append([c, None])
        zeroed.append(i)
    else:
        shutil.copyfile(SRC / "depth" / f"{src:06d}.png", DST / "depth" / f"{i:06d}.png")
        fixed_pairs.append([c, pairs[src][1]])
    source_row.append(src)

for name in ("calib.json", "boxes.json"):
    shutil.copyfile(SRC / name, DST / name)

out = {
    "video_id": sync["video_id"],
    "split": sync["split"],
    "note": f"re-paired by stamp for frames {LO}-{HI}; depth stamp null = no depth available (image is all zeros)",
    "pairs": fixed_pairs,
    "depth_source_row": source_row,
}
(ROOT / "variant/scene_a/sync_fixed.json").write_text(json.dumps(out))
changed = [i for i in range(len(pairs)) if source_row[i] != i]
print(f"changed {len(changed)} frames ({changed[0]}-{changed[-1]}); zeroed depth for {zeroed}")
print(f"wrote {DST} and variant/scene_a/sync_fixed.json")
