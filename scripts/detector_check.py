"""Per-class detector check on the dev scenes' query frames, using query_targets.json + boxes.json.

Geometry-free (detector only): is the queried class detected (score >= min_score) on frames where the target is
visible, and how strongly; does it fire on frames where the target is absent.
Geometry-dependent (needs our lift, so also sensitive to pose/calibration): does a detection of that class land in a
visible target's box (GOAL_MARGIN_M), is the top-scoring one the one that does, and which OTHER classes land there.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import run_part2 as R
from perception.grounding import ConfigPart2, detection_to_3d, frame_depth
from score import GOAL_MARGIN_M, in_box

cfg = ConfigPart2()
det = json.loads((ROOT / "variant/detections.json").read_text())["scenes"]
vis_rows = defaultdict(dict)   # (scene, label) -> {frame: row}
abs_rows = defaultdict(dict)


def lift(scene, f, d, depth, cam):
    p = detection_to_3d(scene, f, d, depth, cam, cfg)
    return None if p is None else p["point_world"]


for name in ("scene_a", "scene_b"):
    sc = R.load_scene(name)
    boxes = {b["uid"]: b for b in sc.boxes}
    tg = json.loads((ROOT / "variant" / name / "query_targets.json").read_text())
    for t in tg.values():
        label = t["label"]
        for f, fr in t["frames"].items():
            dets = [d for d in det[name].get(f, []) if d["score"] >= cfg.min_score]
            cand = sorted((d for d in dets if d["label"] == label), key=lambda d: -d["score"])
            if not fr["visible_targets"]:
                abs_rows[(name, label)][f] = {"n": len(cand), "max": cand[0]["score"] if cand else 0.0}
                continue
            vb = [boxes[u] for u in fr["visible_targets"]]
            depth, cam = frame_depth(sc, int(f), cfg)
            on = []
            for d in cand:
                p = lift(sc, int(f), d, depth, cam)
                on.append(p is not None and any(in_box(p, b, GOAL_MARGIN_M) for b in vb))
            others = Counter()
            for d in dets:
                if d["label"] == label:
                    continue
                p = lift(sc, int(f), d, depth, cam)
                if p is not None and any(in_box(p, b, GOAL_MARGIN_M) for b in vb):
                    others[d["label"]] += 1
            vis_rows[(name, label)][f] = {"detected": bool(cand), "best": cand[0]["score"] if cand else 0.0,
                                          "top_on": bool(on and on[0]), "any_on": any(on), "others": others}

print(f"{'scene':8s} {'class':13s} {'vis frames':>10s} {'detected':>9s} {'mean best':>10s} {'top on tgt':>11s} {'any on tgt':>11s}   other classes landing on the target (count)")
for (name, label), rows in sorted(vis_rows.items()):
    n = len(rows)
    det_n = sum(r["detected"] for r in rows.values())
    mb = np.mean([r["best"] for r in rows.values() if r["detected"]]) if det_n else float("nan")
    oth = sum((r["others"] for r in rows.values()), Counter())
    print(f"{name:8s} {label:13s} {n:10d} {det_n:5d}/{n:<3d} {mb:10.2f} {sum(r['top_on'] for r in rows.values()):6d}/{n:<4d} {sum(r['any_on'] for r in rows.values()):6d}/{n:<4d}   {dict(oth.most_common(4))}")
print("\nabsent frames (target not visible): detections of the queried class above threshold")
for (name, label), rows in sorted(abs_rows.items()):
    print(f"  {name} {label:9s} frames={len(rows)}  frames with a detection={sum(r['n'] > 0 for r in rows.values())}  max score={max(r['max'] for r in rows.values()):.2f}")
