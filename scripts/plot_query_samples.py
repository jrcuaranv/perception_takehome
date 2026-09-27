"""Overlay the query, all 2D detections (target class highlighted) and the surveyed target box on scene_a sample frames."""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from perception.geometry import project, transform
from perception.grounding import nms_2d
from perception.query import parse_query
from perception.scene import Scene

FX = float(sys.argv[sys.argv.index("--fx") + 1]) if "--fx" in sys.argv else None
ARGS = [a for a in sys.argv[1:] if not a.startswith("--") and a != str(FX).rstrip("0").rstrip(".") and a != sys.argv[sys.argv.index("--fx") + 1 if "--fx" in sys.argv else 0]]
NMS = "--nms" in sys.argv
NAME = ARGS[0] if ARGS else "scene_a"
ALL_SAMPLES = {
    "scene_a": [("a01", 975), ("a02", 1015), ("a03", 95), ("a04", 130), ("a05", 450),
                ("a06", 125), ("a07", 960), ("a08", 850), ("a09", 1150), ("a06", 1250)],
    "scene_b": [("b01", 12), ("b02", 100), ("b03", 540), ("b04", 236), ("b05", 1000),
                ("b06", 200), ("b07", 420), ("b08", 932), ("b09", 388), ("b01", 520)],
}
SAMPLES = ALL_SAMPLES[NAME]
TARGET, OTHER, GT = "#e07b00", "#8895a3", "#1e8a4c"

sc = Scene(ROOT / "data" / NAME)
queries = {q["id"]: q for q in json.loads((ROOT / "variant/queries.json").read_text())["scenes"][NAME]}
targets = json.loads((ROOT / f"variant/{NAME}/query_targets.json").read_text())
dets = json.loads((ROOT / "variant/detections.json").read_text())["scenes"][NAME]
boxes = {b["uid"]: b for b in sc.boxes}
RECORDED_FX = sc.depth_cam["fx"]
if FX:
    sc.depth_cam["fx"] = sc.depth_cam["fy"] = FX  # only the projection of the survey boxes uses this; detections are in pixels


def project_box(b, frame, cam=None):
    cam = cam or sc.depth_cam
    c, ax, half = np.array(b["center"]), np.array(b["axes"]), np.array(b["size"]) / 2
    signs = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)])
    corners = c + (signs * half) @ ax
    cam_pts = transform(np.linalg.inv(sc.pose(frame)), corners)
    if (cam_pts[:, 2] <= 0.05).any():
        return None  # partly behind the camera: skip rather than draw a wrong rectangle
    uv, _ = project(cam_pts, cam)
    return uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()


fig, axes = plt.subplots(2, 5, figsize=(22, 9.6))
for ax, (qid, f) in zip(axes.ravel(), SAMPLES):
    q, t = queries[qid], targets[qid]
    p = parse_query(q["text"])
    ax.imshow(sc.rgb(f))
    raw_dets = dets[str(f)]
    frame_dets = nms_2d(raw_dets, 0.5) if NMS else raw_dets
    for d in sorted(frame_dets, key=lambda d: d["label"] == p["target"]):
        x0, y0, x1, y1 = d["box"]
        hit = d["label"] == p["target"]
        ax.add_patch(mp.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=TARGET if hit else OTHER,
                                  lw=2.2 if hit else 0.9, alpha=1 if hit else 0.75))
        ax.text(x0 + 1, y0 + 1, f"{d['label']} {d['score']:.2f}", fontsize=6.5 if not hit else 8, va="top",
                color="white", bbox=dict(fc=TARGET if hit else "#4b5563", ec="none", pad=0.6, alpha=0.85))
    vis = t["frames"][str(f)]["visible_targets"]
    for uid in vis:
        r = project_box(boxes[uid], f)
        if r:
            ax.add_patch(mp.Rectangle((r[0], r[1]), r[2] - r[0], r[3] - r[1], fill=False, ec=GT, lw=2, ls="--"))
    n_t = sum(d["label"] == p["target"] for d in frame_dets)
    ax.set_title(f"{qid}  \"{q['text']}\"  frame {f}\n{t['kind']} | target-class detections: {n_t} | {'kept ' + str(len(frame_dets)) + '/' + str(len(raw_dets)) + ' after NMS | ' if NMS else ''} surveyed targets in view: {len(vis)}",
                 fontsize=9, loc="left")
    ax.set_xlim(0, 256); ax.set_ylim(192, 0); ax.set_xticks([]); ax.set_yticks([])
handles = [mp.Patch(fc="none", ec=TARGET, lw=2, label="detections of the queried class"),
           mp.Patch(fc="none", ec=OTHER, lw=1, label="all other detections"),
           mp.Patch(fc="none", ec=GT, lw=2, ls="--", label="surveyed target box (dev ground truth), projected")]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10, frameon=False)
fig.suptitle(f"{NAME}{f' (survey boxes projected with fx = fy = {FX:g}, recorded {RECORDED_FX:.1f})' if FX else ''}: query, 2D detections{' after class-wise 2D NMS (IoU > 0.5)' if NMS else ''} and surveyed targets on sample frames", fontsize=13)
fig.tight_layout(rect=[0, 0.04, 1, 0.96])
out = ROOT / f"deliverables/plots/part2_{NAME}_samples{'_nms' if NMS else ''}{f'_fx{FX:g}' if FX else ''}.png"
fig.savefig(out, dpi=100)
print(out)


# ---- numbers: pixel distance from each projected survey box centre to the nearest same-class detection centre ----
def offsets(fx):
    cam = dict(sc.depth_cam); cam["fx"] = cam["fy"] = fx
    out = []
    for qid, q in queries.items():
        for f in q["frames"]:
            for uid in targets[qid]["frames"][str(f)]["visible_targets"]:
                r = project_box(boxes[uid], f, cam)
                cs = [((d["box"][0] + d["box"][2]) / 2, (d["box"][1] + d["box"][3]) / 2)
                      for d in dets[str(f)] if d["label"] == boxes[uid]["label"] and d["score"] >= 0.3]
                if r and cs:
                    g = ((r[0] + r[2]) / 2, (r[1] + r[3]) / 2)
                    out.append((min(np.hypot(g[0] - c[0], g[1] - c[1]) for c in cs), f))
    return out

print(f"\n{NAME}: distance (px) between projected survey-box centre and nearest same-class detection (score >= 0.3)")
for fx in sorted({RECORDED_FX if not FX else float(f"{RECORDED_FX:.3f}"), FX or RECORDED_FX, 213.0}):
    o = offsets(fx)
    early = [d for d, f in o if f < 500]; late = [d for d, f in o if f >= 500]
    print(f"  fx={fx:7.2f}: n={len(o):2d} median {np.median([d for d, _ in o]):5.1f}  | frames<500: {np.median(early) if early else float('nan'):5.1f} (n={len(early)})  frames>=500: {np.median(late) if late else float('nan'):5.1f} (n={len(late)})")
