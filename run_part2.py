"""Part 2 driver: variant/queries.json + variant/detections.json -> answers.json.

usage: uv run python run_part2.py [--eval]
--eval prints, for dev scenes, every (query, frame): visible targets, our goal, hit/miss and confidence, plus score_part2.
"""

import argparse
import json
from pathlib import Path

from perception.grounding import ConfigPart2, resolve_query
from perception.query import parse_query
from perception.scene import Scene

ROOT = Path(__file__).resolve().parent
# Part 1 fixes: same frame indices/poses, but scene_a's lagged depth is re-paired and scene_c's frozen depth is zeroed
SCENE_ROOT = {"scene_a": "scene_a_fixed", "scene_c": "scene_c_fixed"}


def load_scene(name):
    return Scene(ROOT / "data" / SCENE_ROOT.get(name, name))


def answer_scene(name, queries, detections, vocabulary, cfg):
    """Every (query, frame) is answered from that frame's detections and depth alone."""
    scene = load_scene(name)
    parsed = {q["id"]: parse_query(q["text"], vocabulary) for q in queries}
    answers, debug = {}, {}
    for q in queries:
        answers[q["id"]], debug[q["id"]] = {}, {}
        for f in q["frames"]:
            goal, conf, dbg = resolve_query(scene, f, parsed[q["id"]], detections.get(str(f), []), cfg)
            answers[q["id"]][str(f)] = {"goal_world": goal, "confidence": round(conf, 4)}
            debug[q["id"]][str(f)] = dbg
    return scene, parsed, answers, debug


def print_eval(scene, queries, parsed, answers, targets, debug):
    from score import GOAL_MARGIN_M, in_box

    print(f"\n{scene.name}")
    for q in queries:
        t = targets[q["id"]]
        for f in q["frames"]:
            a, vis = answers[q["id"]][str(f)], t["frames"][str(f)]["visible_targets"]
            goal = a["goal_world"]
            boxes = [b for b in scene.boxes if b["uid"] in vis]
            hit = None if goal is None else any(in_box(goal, b, GOAL_MARGIN_M) for b in boxes)
            tag = {None: "null", True: "HIT ", False: "miss"}[hit]
            print(f"  {q['id']} {q['text']:32s} f={f:<5d} {t['kind']:9s} visible={len(vis):d}  {tag} conf={a['confidence']:.2f}  {debug[q['id']][str(f)].get('why', '')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "answers.json"))
    args = ap.parse_args()

    cfg = ConfigPart2()
    queries = json.loads((ROOT / "variant/queries.json").read_text())["scenes"]
    det = json.loads((ROOT / "variant/detections.json").read_text())
    all_answers = {}
    for name in queries:
        scene, parsed, answers, debug = answer_scene(name, queries[name], det["scenes"][name], set(det["vocabulary"]), cfg)
        all_answers[name] = answers
        n = sum(a["goal_world"] is not None for qa in answers.values() for a in qa.values())
        total = sum(len(qa) for qa in answers.values())
        print(f"{name}: answered {n}/{total}")
        tp = ROOT / "variant" / name / "query_targets.json"
        if args.eval and tp.exists():
            print_eval(scene, queries[name], parsed, answers, json.loads(tp.read_text()), debug)
    Path(args.out).write_text(json.dumps(all_answers, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
