"""Part 3 driver: variant/detections.json -> instances.json (one instance per physical object per scene).

usage: uv run python run_part3.py [--eval] [--out instances.json]
--eval also scores the dev scenes (same numbers as score.py) and prints instances per class for every scene.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from perception.mapping import ConfigPart3, build_map, finalize, lift_frame_candidates
from perception.scene import Scene
from run_part2 import load_scene  # uses scene_a_fixed / scene_c_fixed (Part 1 sync fixes)

ROOT = Path(__file__).resolve().parent


def map_scene(name, detections, cfg):
    scene = load_scene(name)
    cands = lift_frame_candidates(scene, detections, cfg)
    return finalize(build_map(cands, cfg), cfg), sum(len(c) for c in cands.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "instances.json"))
    args = ap.parse_args()

    cfg = ConfigPart3()
    det = json.loads((ROOT / "variant/detections.json").read_text())["scenes"]
    result = {}
    for name in det:
        result[name], n_cands = map_scene(name, det[name], cfg)
        print(f"{name}: {n_cands} lifted detections -> {len(result[name])} instances {dict(Counter(i['label'] for i in result[name]))}")
        if args.eval and (ROOT / "data" / name / "boxes.json").exists():
            from score import score_part3

            print("   ", json.dumps(score_part3(Scene(ROOT / "data" / name), det[name], result[name])))
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
