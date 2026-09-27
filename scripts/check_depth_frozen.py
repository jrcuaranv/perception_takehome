"""Are depth images identical inside scene_c's stalled block? Compares raw depth PNG pixels (and sync stamps) frame to frame."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from perception.scene import Scene

name = sys.argv[1] if len(sys.argv) > 1 else "scene_c"
sc = Scene(ROOT / "data" / name)
pairs = json.loads((ROOT / f"variant/{name}/sync.json").read_text())["pairs"]
n = len(sc)

raw = [sc.depth_raw(i) for i in range(n)]
hashes = [hashlib.sha256(r.tobytes()).hexdigest() for r in raw]
stamps = [p[1] for p in pairs]

# Runs of consecutive frames whose depth is pixel-identical to the previous frame
same = [hashes[i] == hashes[i - 1] for i in range(1, n)]
runs, i = [], 0
while i < len(same):
    if same[i]:
        j = i
        while j < len(same) and same[j]:
            j += 1
        runs.append((i, j))  # frames i..j all share one depth image
        i = j
    else:
        i += 1

print(f"{name}: {n} frames, {len(set(hashes))} distinct depth images, {len(set(stamps))} distinct depth stamps")
print(f"runs of pixel-identical consecutive depth (first_frame, last_frame, length):")
for a, b in runs:
    print(f"  {a}-{b}  len {b - a + 1}  depth stamp {stamps[a]}  colour stamps {pairs[a][0]}..{pairs[b][0]}")

lo, hi = 58, 105
blk = hashes[lo : hi + 1]
print(f"\nblock {lo}-{hi}: {len(blk)} frames, {len(set(blk))} distinct depth image(s), "
      f"{len(set(stamps[lo:hi + 1]))} distinct depth stamp(s)")
ref = raw[lo].astype(np.int32)
maxdiff = max(int(np.abs(raw[k].astype(np.int32) - ref).max()) for k in range(lo, hi + 1))
print(f"max |pixel difference| vs frame {lo} across block: {maxdiff} raw units")

# Context: how different are ordinary neighbouring frames outside any run?
outside = [k for k in range(1, n) if not same[k - 1]][:200]
diffs = [float(np.mean(raw[k] != raw[k - 1])) for k in outside]
print(f"outside runs: median fraction of pixels that change between consecutive frames = {np.median(diffs):.3f}")
