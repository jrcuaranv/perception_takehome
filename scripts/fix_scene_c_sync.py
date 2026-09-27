"""Build data/scene_c_fixed/ and variant/scene_c/sync_fixed.json without touching scene_c.

Frames 58-105 all carry the single depth image of frame 57 (depth frozen while colour advanced). If some depth row
holds the correct stamp for a colour frame we re-pair it; otherwise its depth is written as all zeros (0 = no
return) so frame indices, poses and detections stay aligned. Everything else is copied as recorded.
"""
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC, DST = ROOT / "data/scene_c", ROOT / "data/scene_c_fixed"
LO, HI = 58, 105

sync = json.loads((ROOT / "variant/scene_c/sync.json").read_text())
pairs = sync["pairs"]
row_of = {}
for r, (_, d) in enumerate(pairs):
    row_of.setdefault(d, []).append(r)

if DST.exists():
    raise SystemExit(f"{DST} exists; move or delete it to rebuild")
for sub in ("rgb", "depth"):
    (DST / sub).mkdir(parents=True)

fixed_pairs, source_row, zeroed, repaired = [], [], [], []
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
        if src != i:
            repaired.append(i)
    source_row.append(src)

for name in ("calib.json",):
    shutil.copyfile(SRC / name, DST / name)

out = {
    "video_id": sync["video_id"],
    "split": sync["split"],
    "note": f"frames {LO}-{HI}: stale/frozen depth removed; depth stamp null = no depth available (image is all zeros)",
    "pairs": fixed_pairs,
    "depth_source_row": source_row,
}
(ROOT / "variant/scene_c/sync_fixed.json").write_text(json.dumps(out))
print(f"re-paired {len(repaired)} frames; zeroed depth for {len(zeroed)} frames ({zeroed[0]}-{zeroed[-1]})" if zeroed else "nothing zeroed")
print(f"wrote {DST} and variant/scene_c/sync_fixed.json")
