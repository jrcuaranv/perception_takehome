"""Part 2: lift 2D detections to world-frame goals, pick among candidates, and say how much to believe them.

Pipeline (see run_part2.py):
  frame_depth -> nms_2d -> mask_to_depth -> detection_to_3d -> dedupe_frame_candidates -> resolve_query
Single-frame only. lift_scene_candidates/build_landmarks (scene-level memory) are kept for Part 3, not used here.
"""

from dataclasses import dataclass, field

import numpy as np

from perception.geometry import aligned_depth_to_color, transform


@dataclass
class ConfigPart2:
    depth_mode: str = "pixel_aligned"  # or "warped" (aligned_depth_to_color with the recorded calibration)
    depth_scale_override: dict = field(default_factory=dict)  # {scene: scale_m}; empty = trust calib.json
    min_score: float = 0.15  # detections below this are not used at all
    min_z: float = 0.3
    max_z: float = 6.0
    central: float = 0.5  # fraction of the box (per side) whose depth is used
    percentile: float = 20  # near-surface depth percentile inside that patch
    band: float = 0.06  # metres around that depth kept as the object surface
    nms_iou: float = 0.5  # class-wise 2D NMS threshold
    dedupe_radius: float = 0.3  # same-frame, same-label detections closer than this are one object
    # landmark_radius: float = 0.5  # cross-frame clustering radius
    min_support: int = 3  # a landmark needs this many distinct frames or detections on it are ignored


# ---------------------------------------------------------------- depth --------------------------------------------


def frame_depth(scene, frame, cfg):
    """(depth in metres on the colour pixel grid, camera dict whose intrinsics unproject it).

    pixel_aligned: depth pixel (u, v) is taken to be colour pixel (u, v) (what Part 1 supports empirically).
    warped: warp with the recorded T_color_depth and colour intrinsics (only worth using with a corrected calibration)."""
    scale = cfg.depth_scale_override.get(scene.name, scene.depth_cam["scale_m"])
    raw = scene.depth_raw(frame)
    if cfg.depth_mode == "warped":
        return aligned_depth_to_color(raw, scene.depth_cam, scene.color, scene.T_color_depth, scale_m=scale), scene.color
    return raw.astype(np.float32) * scale, scene.depth_cam


def mask_to_depth(depth_m, box, central=0.5, percentile=20, band=0.06, min_valid=10):
    """Representative depth for a 2D box: the near-surface percentile of the valid depth in the box's central patch.

    Returns (z, (rows, cols) of the pixels within `band` of z, info) or None if there is not enough depth.
    The percentile picks the object's near surface instead of the wall behind it; info feeds the confidence."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hw, hh = (x1 - x0) * central / 2, (y1 - y0) * central / 2
    h, w = depth_m.shape
    r0, r1 = int(max(0, cy - hh)), int(min(h, cy + hh + 1))
    c0, c1 = int(max(0, cx - hw)), int(min(w, cx + hw + 1))
    patch = depth_m[r0:r1, c0:c1]
    valid = patch > 0
    n_valid = int(valid.sum())
    if n_valid < min_valid:
        return None
    z = float(np.percentile(patch[valid], percentile))
    inband = valid & (np.abs(patch - z) < band)
    if not inband.any():
        return None
    rows, cols = np.nonzero(inband)
    info = {"valid_frac": n_valid / patch.size, "band_frac": float(inband.sum()) / n_valid, "n_band": int(inband.sum())}
    return z, (rows + r0, cols + c0), info


def detection_to_3d(scene, frame, det, depth_m, cam, cfg):
    """World point for one detection (mean of the surface band, through the frame's pose) plus quality info, or None."""
    res = mask_to_depth(depth_m, det["box"], cfg.central, cfg.percentile, cfg.band)
    if res is None:
        return None
    z, (rows, cols), info = res
    if not (cfg.min_z <= z <= cfg.max_z):
        return None
    zs = depth_m[rows, cols].astype(np.float64)
    pts = np.stack([(cols - cam["cx"]) * zs / cam["fx"], (rows - cam["cy"]) * zs / cam["fy"], zs], axis=1)
    point = transform(scene.pose(frame), pts.mean(axis=0, keepdims=True))[0]
    return {"frame": frame, "label": det["label"], "score": float(det["score"]), "box": det["box"],
            "point_world": point, "z": z, **info}


def lift_scene_candidates(scene, detections, labels, cfg):
    """{frame: [candidate, ...]} for every detection of the given labels that lifts to 3D."""
    out = {}
    for f_str, dets in detections.items():
        f = int(f_str)
        picked = [(k, d) for k, d in enumerate(dets) if d["label"] in labels and d["score"] >= cfg.min_score]
        if not picked:
            continue
        depth, cam = frame_depth(scene, f, cfg)
        cands = []
        for k, d in picked:
            c = detection_to_3d(scene, f, d, depth, cam, cfg)
            if c is not None:
                c["det_index"] = k
                cands.append(c)
        out[f] = dedupe_frame_candidates(cands, cfg.dedupe_radius)
    return out


def dedupe_frame_candidates(cands, radius):
    """3D non-maximum suppression: several boxes on one object in one frame collapse to the best-scoring one."""
    kept = []
    for c in sorted(cands, key=lambda c: -c["score"]):
        if not any(k["label"] == c["label"] and np.linalg.norm(k["point_world"] - c["point_world"]) < radius for k in kept):
            kept.append(c)
    return kept


# ------------------------------------------------------------- scene memory --------------------------------------


@dataclass
class Landmark:
    id: int
    label: str
    center: np.ndarray
    n_frames: int
    n_obs: int
    mean_score: float


def build_landmarks(cands_by_frame, radius):
    """Cluster candidates across frames into world landmarks, per label. Assigns c['landmark'] (index into the
    returned {label: [Landmark]} list, or None) on every candidate.

    Greedy by score (best detections seed clusters), then one reassignment pass against the final centres.
    This is what lets a query be answered with the same instance from every viewpoint, and lets 'nearest the stove'
    use a stove that is not in the current frame."""
    flat = [c for cs in cands_by_frame.values() for c in cs]
    landmarks = {}
    for label in sorted({c["label"] for c in flat}):
        cs = sorted((c for c in flat if c["label"] == label), key=lambda c: -c["score"])
        centers, weights = [], []
        for c in cs:
            p = c["point_world"]
            if centers:
                d = np.linalg.norm(np.array(centers) - p, axis=1)
                j = int(d.argmin())
                if d[j] < radius:
                    w = weights[j]
                    centers[j] = (centers[j] * w + p * c["score"]) / (w + c["score"])
                    weights[j] = w + c["score"]
                    continue
            centers.append(p.copy())
            weights.append(c["score"])
        centers = np.array(centers)
        assign = [int(np.linalg.norm(centers - c["point_world"], axis=1).argmin()) for c in cs]
        lms = []
        for j in range(len(centers)):
            members = [c for c, a in zip(cs, assign) if a == j and np.linalg.norm(centers[j] - c["point_world"]) < radius]
            if not members:
                continue
            w = np.array([m["score"] for m in members])
            center = (np.array([m["point_world"] for m in members]) * w[:, None]).sum(0) / w.sum()
            lms.append((center, members))
        landmarks[label] = []
        for i, (center, members) in enumerate(lms):
            landmarks[label].append(Landmark(i, label, center, len({m["frame"] for m in members}), len(members),
                                             float(np.mean([m["score"] for m in members]))))
            for m in members:
                m["landmark"] = i
        for c in cs:
            c.setdefault("landmark", None)
    return landmarks


# ------------------------------------------------------- query answering (single frame) -------------------------

NULL_CONFIDENCE = 0.1  # README's example for a null goal; score.py does not use confidence on unanswered frames
QUALIFIER_UNVERIFIED = 0.7  # multiplier when only one target is in view: 'nearest the Y' cannot be checked
REFERENCE_MISSING = 0.5  # multiplier when the reference is not detected: the qualifier is ignored


def lift_class_detections(scene, frame, dets, label, depth, cam, cfg):
    """(candidates, n_boxes): one class's detections in one frame, lifted to 3D, best score first.

    score filter -> class-wise 2D NMS -> lift each box (boxes without valid depth drop out, so the next best
    is used) -> 3D dedupe (nested part/whole boxes that survive 2D NMS collapse to one object).
    n_boxes counts the boxes before lifting, to tell 'not detected' from 'detected but no depth'."""
    boxes = [d for d in nms_2d(dets, cfg.nms_iou) if d["label"] == label and d["score"] >= cfg.min_score]
    cands = []
    for d in boxes:
        c = detection_to_3d(scene, frame, d, depth, cam, cfg)
        if c is not None:
            c["det_index"] = d["det_index"]
            cands.append(c)
    return dedupe_frame_candidates(cands, cfg.dedupe_radius), len(boxes)


def estimate_confidence(target_score, qualified=False, reference_score=None, n_targets=1):
    """Heuristic P(goal matches the instruction).
    Unqualified: the detector score of the chosen box.
    Qualified: that score, times how much to trust the qualifier:
      reference not detected  -> x REFERENCE_MISSING (qualifier ignored)
      one target in view      -> x (0.5 + 0.5 * reference_score) x QUALIFIER_UNVERIFIED (nothing to compare against)
      several targets         -> x (0.5 + 0.5 * reference_score)"""
    p = target_score
    if qualified:
        if reference_score is None:
            p *= REFERENCE_MISSING
        else:
            if n_targets == 1:
                p = p*(0.5 + 0.5 * reference_score)*QUALIFIER_UNVERIFIED
            else:
                p = p*(0.5 + 0.5 * reference_score)
            
    return float(np.clip(p, 0.02, 0.98))


def resolve_query(scene, frame, parsed, dets, cfg):
    """(goal_world | None, confidence, debug) for one query from ONE frame's detections; no other frame is used.

    Unqualified: the best-scoring target after NMS/dedupe. Qualified ('X nearest the Y'): the best-scoring reference
    after NMS/dedupe, then the target closest to it in 3D. Not detected -> null. Reference not detected -> fall
    back to the best target with reduced confidence."""
    depth, cam = frame_depth(scene, frame, cfg)
    targets, n_boxes = lift_class_detections(scene, frame, dets, parsed["target"], depth, cam, cfg)
    if not targets:
        return None, NULL_CONFIDENCE, {"why": "no target detected" if n_boxes == 0 else "target detected but no valid depth"}
    dbg = {"n_targets": len(targets)}
    best, ref_score = targets[0], None
    qualified = parsed["relation"] is not None
    if qualified:
        refs, _ = lift_class_detections(scene, frame, dets, parsed["reference"], depth, cam, cfg)
        if not refs:
            dbg["why"] = "reference not detected; qualifier ignored"
        else:
            ref, ref_score = refs[0], refs[0]["score"]
            best = min(targets, key=lambda t: np.linalg.norm(t["point_world"] - ref["point_world"]))
    conf = estimate_confidence(best["score"], qualified, ref_score, len(targets))
    return [float(x) for x in best["point_world"]], conf, {**dbg, "score": best["score"], "det_index": best["det_index"]}


# ------------------------------------------------------------ 2D detection cleanup --------------------------------


def iou_2d(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def nms_2d(dets, iou_thr=0.5):
    """Class-wise 2D non-maximum suppression on one frame's detections ({label, score, box}).

    Within each label, keep the highest-scoring box and drop any box overlapping a kept one by IoU > iou_thr.
    Boxes of different labels never suppress each other (an oven and a stove on one appliance both survive).
    Returns the kept detections, each with 'det_index' = its index in the input list."""
    kept = []
    for k in sorted(range(len(dets)), key=lambda k: -dets[k]["score"]):
        d = dets[k]
        if not any(x["label"] == d["label"] and iou_2d(x["box"], d["box"]) > iou_thr for x in kept):
            kept.append({**d, "det_index": k})
    return kept
