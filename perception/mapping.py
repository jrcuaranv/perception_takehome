"""Part 3: build a persistent object map from per-frame detections, one instance per physical object.

Pipeline (see run_part3.py):
  lift_frame_candidates -> build_map (frame by frame: associate, merge or create) -> merge_close -> finalize
Each instance keeps a running weighted centre and a Beta(alpha, beta) pseudo-count for the semantic confidence.
Association uses a class-size prior: a new detection joins the nearest same-label instance whose centre is within the
label's radius (x gate_scale), one detection per instance per frame.
"""

from dataclasses import dataclass, field

import numpy as np

from perception.grounding import ConfigPart2, dedupe_frame_candidates, detection_to_3d, frame_depth, nms_2d

# Rough half-size of each class (m): used to gate association and to push a lifted surface point back to the centre.
OBJECT_RADIUS = {
    "cabinet": 0.30,
    "refrigerator": 0.50,
    "shelf": 0.60,
    "stove": 0.45,
    "bed": 1.10,
    "sink": 0.40,
    "washer": 0.45,
    "toilet": 0.40,
    "bathtub": 0.90,
    "oven": 0.45,
    "dishwasher": 0.45,
    "fireplace": 0.5,
    "stool": 0.30,
    "chair": 0.40,
    "table": 0.5,
    "tv_monitor": 0.45,
    "sofa": 1.00,
}


@dataclass
class ConfigPart3:
    lift: ConfigPart2 = field(default_factory=ConfigPart2)  # depth mode, min_score, NMS and dedupe settings
    gate_scale: float = 1.5  # association gate = gate_scale * OBJECT_RADIUS[label]
    push_back: bool = True  # move the lifted near-surface point one radius further along the viewing ray
    range_power: float = 0.0  # position weight = score / max(z, 1) ** range_power (0 = score only)
    prior_alpha: float = 1.0
    prior_beta: float = 1.0
    final_merge: bool = True  # after the pass, merge same-label instances that ended up within the gate
    coview_frac: float = 1.0  # same-frame, same-label detections whose centres are closer than coview_frac * radius are
    #                           treated as duplicate boxes on one object (0 = every surviving detection is a different object)
    coview_cannot_link: bool = False  # final merge refuses instances that were both seen in one frame
    min_frames: int = 3  # distinct frames an instance must be seen in to be reported
    min_confidence: float = 0.3  # semantic confidence (alpha / (alpha + beta)) needed to be reported
    min_score_sum: float = 5.0  # summed detector score over the instance's observations needed to be reported


# ------------------------------------------------------------------- lifting -------------------------------------


def lift_frame_candidates(scene, detections, cfg):
    """{frame: [candidate, ...]}: every usable detection lifted to a world-frame surface point.

    NMS -> score filter -> lift (near-surface point, no depth -> dropped) -> same-frame 3D dedupe. Each candidate carries
    its raw det_index (index in that frame's list in detections.json) and the camera position for the push-back."""
    out = {}
    for f_str, dets in detections.items():
        f = int(f_str)
        kept = [d for d in nms_2d(dets, cfg.lift.nms_iou) if d["score"] >= cfg.lift.min_score and d["label"] in OBJECT_RADIUS]
        if not kept:
            continue
        depth, cam = frame_depth(scene, f, cfg.lift)
        cands = []
        for d in kept:
            c = detection_to_3d(scene, f, d, depth, cam, cfg.lift)
            if c is not None:
                c["det_index"] = d["det_index"]
                cands.append(c)
        cands = dedupe_frame_candidates(cands, cfg.lift.dedupe_radius)
        cam_pos = scene.pose(f)[:3, 3]
        for c in cands:
            c["cam_pos"] = cam_pos
        if cands:
            out[f] = cands
    return out


def object_center(cand, cfg):
    """Estimated object centre: the lifted surface point, pushed one class radius further along the viewing ray, so views
    from opposite sides of a large object (a sofa) agree instead of sitting one object-width apart."""
    p = cand["point_world"]
    if not cfg.push_back:
        return p
    ray = p - cand["cam_pos"]
    n = np.linalg.norm(ray)
    return p if n < 1e-6 else p + OBJECT_RADIUS[cand["label"]] * ray / n


# ------------------------------------------------------------------- the map -------------------------------------


def _new_instance(iid, cand, center, w, cfg):
    s = cand["score"]
    return {"id": iid, "label": cand["label"], "center": center.copy(), "position_weight": w,
            "alpha": cfg.prior_alpha + s, "beta": cfg.prior_beta + 1 - s,
            "observations": [[cand["frame"], cand["det_index"]]], "frames": {cand["frame"]}}


def _merge_observation(inst, cand, center, w):
    W = inst["position_weight"]
    inst["center"] = (W * inst["center"] + w * center) / (W + w)
    inst["position_weight"] = W + w
    inst["alpha"] += cand["score"]
    inst["beta"] += 1 - cand["score"]
    inst["observations"].append([cand["frame"], cand["det_index"]])
    inst["frames"].add(cand["frame"])


def _weight(cand, cfg):
    return cand["score"] / max(cand["z"], 1.0) ** cfg.range_power


def build_map(cands_by_frame, cfg):
    """Process frames in order. Per frame: candidates are matched one-to-one to same-label instances within the gate
    (nearest pairs first); unmatched candidates start new instances, so two same-class detections in one frame can
    never end up in the same instance."""
    instances, next_id = [], 1
    for f in sorted(cands_by_frame):
        cands = sorted(cands_by_frame[f], key=lambda c: -c["score"])
        centers = [object_center(c, cfg) for c in cands]
        dups = {}  # index of a kept candidate -> redundant same-object boxes in this frame (observations only)
        if cfg.coview_frac > 0:
            kept = []
            for j, c in enumerate(cands):
                r = cfg.coview_frac * OBJECT_RADIUS[c["label"]]
                host = next((k for k in kept if cands[k]["label"] == c["label"] and np.linalg.norm(centers[k] - centers[j]) < r), None)
                if host is None:
                    kept.append(j)
                    dups[j] = []
                else:
                    dups[host].append(c)
            remap = {old: new for new, old in enumerate(kept)}
            dups = {remap[k]: v for k, v in dups.items()}
            cands, centers = [cands[k] for k in kept], [centers[k] for k in kept]
        pairs = []
        for j, c in enumerate(cands):
            gate = cfg.gate_scale * OBJECT_RADIUS[c["label"]]
            for i, inst in enumerate(instances):
                if inst["label"] == c["label"]:
                    d = float(np.linalg.norm(inst["center"] - centers[j]))
                    if d <= gate:
                        pairs.append((d, j, i))
        used_c, used_i = set(), set()
        for d, j, i in sorted(pairs):
            if j not in used_c and i not in used_i:
                _merge_observation(instances[i], cands[j], centers[j], _weight(cands[j], cfg))
                instances[i]["observations"] += [[d["frame"], d["det_index"]] for d in dups.get(j, [])]
                used_c.add(j)
                used_i.add(i)
        for j, c in enumerate(cands):
            if j not in used_c:
                instances.append(_new_instance(next_id, c, centers[j], _weight(c, cfg), cfg))
                instances[-1]["observations"] += [[d["frame"], d["det_index"]] for d in dups.get(j, [])]
                next_id += 1
    return merge_close(instances, cfg) if cfg.final_merge else instances


def merge_close(instances, cfg):
    """Repair early splits: merge the closest same-label pair within the gate, repeat. Two instances that were both
    seen in one frame are provably different objects (one detection per instance per frame), so they never merge."""
    instances = list(instances)
    while True:
        best = None
        for i in range(len(instances)):
            for j in range(i + 1, len(instances)):
                a, b = instances[i], instances[j]
                if a["label"] != b["label"] or (cfg.coview_cannot_link and a["frames"] & b["frames"]):
                    continue
                d = float(np.linalg.norm(a["center"] - b["center"]))
                if d <= cfg.gate_scale * OBJECT_RADIUS[a["label"]] and (best is None or d < best[0]):
                    best = (d, i, j)
        if best is None:
            return instances
        _, i, j = best
        a, b = instances[i], instances[j]
        W = a["position_weight"] + b["position_weight"]
        a["center"] = (a["position_weight"] * a["center"] + b["position_weight"] * b["center"]) / W
        a["position_weight"] = W
        a["alpha"] += b["alpha"] - cfg.prior_alpha
        a["beta"] += b["beta"] - cfg.prior_beta
        a["observations"] += b["observations"]
        a["frames"] |= b["frames"]
        a["id"] = min(a["id"], b["id"])
        del instances[j]


def finalize(instances, cfg):
    """Keep instances seen in >= min_frames frames with semantic confidence >= min_confidence; instances.json format."""
    out = []
    for inst in sorted(instances, key=lambda i: i["id"]):
        conf = inst["alpha"] / (inst["alpha"] + inst["beta"])
        score_sum = inst["alpha"] - cfg.prior_alpha
        if len(inst["frames"]) < cfg.min_frames or conf < cfg.min_confidence or score_sum < cfg.min_score_sum:
            continue
        out.append({"id": inst["id"], "label": inst["label"],
                    "center_world": [float(x) for x in inst["center"]],
                    "observations": sorted([int(f), int(k)] for f, k in inst["observations"]),
                    "semantic_confidence": round(conf, 4), "alpha": round(inst["alpha"], 4), "beta": round(inst["beta"], 4),
                    "position_weight": round(inst["position_weight"], 4), "n_frames": len(inst["frames"])})
    return out
