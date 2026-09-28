"""Same overlay as plot_query_samples.py (query, all 2D detections with the target class highlighted, and the
surveyed target box) but on a configurable grid -- default 3x3, no NMS. Kept as a separate script so the original
2x5 (with/without NMS, with the fx sweep) stays untouched.

usage: uv run python scripts/plot_query_samples_grid.py [scene_a scene_b ...] [--nms] [--rows 3] [--cols 3]
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.patches as mp
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from perception.geometry import project, transform
from perception.grounding import nms_2d
from perception.query import parse_query
from perception.scene import Scene

ALL_SAMPLES = {
    "scene_a": [("a01", 975), ("a02", 1015), ("a03", 95), ("a04", 130), ("a05", 450),
                ("a06", 125), ("a07", 960), ("a08", 850), ("a09", 1150)],
    "scene_b": [("b01", 12), ("b02", 100), ("b03", 540), ("b04", 236), ("b05", 1000),
                ("b06", 200), ("b07", 420), ("b08", 932), ("b09", 388)],
}
TARGET, OTHER, GT = "#e07b00", "#8895a3", "#1e8a4c"


def project_box(sc, b, frame, cam):
    c, ax, half = np.array(b["center"]), np.array(b["axes"]), np.array(b["size"]) / 2
    signs = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)])
    corners = c + (signs * half) @ ax
    cam_pts = transform(np.linalg.inv(sc.pose(frame)), corners)
    if (cam_pts[:, 2] <= 0.05).any():
        return None  # partly behind the camera: skip rather than draw a wrong rectangle
    uv, _ = project(cam_pts, cam)
    return uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()


def plot_scene(name, rows, cols, nms):
    sc = Scene(ROOT / "data" / name)
    queries = {q["id"]: q for q in json.loads((ROOT / "variant/queries.json").read_text())["scenes"][name]}
    targets = json.loads((ROOT / f"variant/{name}/query_targets.json").read_text())
    dets = json.loads((ROOT / "variant/detections.json").read_text())["scenes"][name]
    boxes = {b["uid"]: b for b in sc.boxes}
    samples = ALL_SAMPLES[name][: rows * cols]

    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 4.3 * rows))
    for ax, (qid, f) in zip(np.atleast_1d(axes).ravel(), samples):
        q, t = queries[qid], targets[qid]
        p = parse_query(q["text"])
        ax.imshow(sc.rgb(f))
        raw_dets = dets[str(f)]
        frame_dets = nms_2d(raw_dets, 0.5) if nms else raw_dets
        for d in sorted(frame_dets, key=lambda d: d["label"] == p["target"]):
            x0, y0, x1, y1 = d["box"]
            hit = d["label"] == p["target"]
            ax.add_patch(mp.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=TARGET if hit else OTHER,
                                      lw=2.2 if hit else 0.9, alpha=1 if hit else 0.75))
            ax.text(x0 + 1, y0 + 1, f"{d['label']} {d['score']:.2f}", fontsize=6.5 if not hit else 8, va="top",
                    color="white", bbox=dict(fc=TARGET if hit else "#4b5563", ec="none", pad=0.6, alpha=0.85))
        vis = t["frames"][str(f)]["visible_targets"]
        for uid in vis:
            r = project_box(sc, boxes[uid], f, sc.depth_cam)
            if r:
                ax.add_patch(mp.Rectangle((r[0], r[1]), r[2] - r[0], r[3] - r[1], fill=False, ec=GT, lw=2, ls="--"))
        n_t = sum(d["label"] == p["target"] for d in frame_dets)
        nms_tag = f"kept {len(frame_dets)}/{len(raw_dets)} after NMS | " if nms else ""
        ax.set_title(f"{qid}  \"{q['text']}\"  frame {f}\n{t['kind']} | target-class detections: {n_t} | {nms_tag}surveyed targets in view: {len(vis)}",
                     fontsize=9, loc="left")
        ax.set_xlim(0, 256); ax.set_ylim(192, 0); ax.set_xticks([]); ax.set_yticks([])
    for ax in np.atleast_1d(axes).ravel()[len(samples):]:
        ax.axis("off")
    handles = [mp.Patch(fc="none", ec=TARGET, lw=2, label="detections of the queried class"),
               mp.Patch(fc="none", ec=OTHER, lw=1, label="all other detections"),
               mp.Patch(fc="none", ec=GT, lw=2, ls="--", label="surveyed target box (dev ground truth), projected")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10, frameon=False)
    fig.suptitle(f"{name}: query, 2D detections{' after class-wise 2D NMS (IoU > 0.5)' if nms else ' (raw, no NMS)'} and surveyed targets on sample frames", fontsize=13)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    out = ROOT / f"deliverables/plots/part2_{name}_samples_{rows}x{cols}{'_nms' if nms else '_raw'}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*", default=["scene_a", "scene_b"])
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--nms", action="store_true", help="apply class-wise 2D NMS before plotting (default: raw detections)")
    args = ap.parse_args()
    for name in args.scenes:
        plot_scene(name, args.rows, args.cols, args.nms)


if __name__ == "__main__":
    main()
