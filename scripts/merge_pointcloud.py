"""Merge valid RGB-D frames of a scene into one coloured world-frame point cloud (.ply).

A frame is valid when
  * its sync entry has a depth stamp (not null) and |depth_stamp - colour_stamp| <= --max-offset seconds, and
  * its depth image has at least --min-pts returns.
Sync entries come from variant/<scene>/sync_fixed.json for *_fixed scenes, else sync.json.

usage: merge_pointcloud.py [SCENE_DIR] [--step N] [--voxel M] [--out FILE] [--max-offset S] [--zmin M] [--zmax M]
  default: data/scene_a_fixed, every 5th frame, 5 cm voxels, deliverables/plots/<scene>.ply
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from perception.geometry import project, transform, unproject
from perception.scene import Scene


def valid_frames(scene, sync_path, max_offset, min_pts):
    pairs = json.loads(sync_path.read_text())["pairs"]
    keep, why = [], {"no depth stamp": 0, "stamp offset": 0, "empty depth": 0}
    for i, (c, d) in enumerate(pairs):
        if d is None:
            why["no depth stamp"] += 1
        elif abs(float(d) - float(c)) > max_offset:
            why["stamp offset"] += 1
        elif (scene.depth_raw(i) > 0).sum() < min_pts:
            why["empty depth"] += 1
        else:
            keep.append(i)
    return keep, why


def frame_points(scene, i, zmin, zmax):
    """World-frame points and RGB colours for frame i."""
    pts_d, _ = unproject(scene.depth_m(i), scene.depth_cam)
    pts_c = transform(scene.T_color_depth, pts_d)  # depth sensor -> colour camera
    uv, z = project(pts_c, scene.color)
    u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
    h, w = scene.color["height"], scene.color["width"]
    ok = (z > zmin) & (z < zmax) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    rgb = scene.rgb(i)[v[ok], u[ok]]
    return transform(scene.pose(i), pts_c[ok]), rgb


def voxel_downsample(xyz, rgb, voxel):
    """Keep one point (mean position and colour) per voxel."""
    key = np.floor(xyz / voxel).astype(np.int64)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    out_xyz = np.zeros((len(cnt), 3))
    out_rgb = np.zeros((len(cnt), 3))
    for k in range(3):
        out_xyz[:, k] = np.bincount(inv, xyz[:, k]) / cnt
        out_rgb[:, k] = np.bincount(inv, rgb[:, k].astype(np.float64)) / cnt
    return out_xyz, np.round(out_rgb).astype(np.uint8)


def write_ply(path, xyz, rgb):
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(xyz)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    rec = np.empty(len(xyz), dtype=[("p", "<f4", 3), ("c", "u1", 3)])
    rec["p"], rec["c"] = xyz.astype(np.float32), rgb
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(rec.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir", nargs="?", default="data/scene_a_fixed")
    ap.add_argument("--step", type=int, default=5, help="use every Nth valid frame")
    ap.add_argument("--voxel", type=float, default=0.05, help="voxel size in metres (0 = no downsampling)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-offset", type=float, default=0.05, help="max |depth - colour| stamp gap in seconds")
    ap.add_argument("--min-pts", type=int, default=500, help="min depth returns for a frame to count")
    ap.add_argument("--zmin", type=float, default=0.3)
    ap.add_argument("--zmax", type=float, default=4.5)
    a = ap.parse_args()

    root = Path(a.scene_dir)
    root = root if root.is_absolute() else ROOT / root
    scene = Scene(root)
    fixed = scene.name.endswith("_fixed")
    base = scene.name.removesuffix("_fixed")
    sync_path = ROOT / "variant" / base / ("sync_fixed.json" if fixed else "sync.json")

    keep, why = valid_frames(scene, sync_path, a.max_offset, a.min_pts)
    print(f"{scene.name}: {len(keep)}/{len(scene)} valid frames; dropped: {why}")
    use = keep[:: a.step]

    xyz, rgb = [], []
    for i in use:
        p, c = frame_points(scene, i, a.zmin, a.zmax)
        xyz.append(p)
        rgb.append(c)
    xyz, rgb = np.concatenate(xyz), np.concatenate(rgb)
    print(f"merged {len(use)} frames -> {len(xyz):,} points")
    if a.voxel > 0:
        xyz, rgb = voxel_downsample(xyz, rgb, a.voxel)
        print(f"voxel {a.voxel} m -> {len(xyz):,} points")

    out = Path(a.out) if a.out else ROOT / "deliverables/plots" / f"{scene.name}.ply"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_ply(out, xyz, rgb)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
