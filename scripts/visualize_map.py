"""3D view of surveyed objects (ground truth) vs the objects our Part 3 map found, colour-coded by label.

  - point cloud: faded grey context (from scripts/merge_pointcloud.py); ceiling points above --ceiling m are dropped
  - surveyed boxes (boxes.json): coloured wireframe boxes
  - estimated instances (instances.json): spheres in the label colour
      big sphere   = centre lies in a same-label surveyed box (+0.35 m, the score.py rule)
      small sphere = no matching surveyed box (a "false instance")

Standalone (numpy, matplotlib, open3d only); run in an env that has open3d, e.g.
  DISPLAY=:1 conda run -n gs3lam python scripts/visualize_map.py [scene_a scene_b] [--show]
Writes deliverables/plots/map_<scene>.png (oblique + top-down); --no-cloud draws only the objects -> map_<scene>_objects.png. --show opens an interactive Open3D window instead.

Workarounds for gs3lam (open3d 0.16 with numpy 2.2): calls that pass vectors/matrices to open3d (paint_uniform_color,
translate/rotate, ViewControl.set_front/lookat/up, background_color) or read its arrays back (np.asarray(mesh.vertices))
segfault. So meshes are built from numpy arrays, the ply is read with numpy, and the scene is rotated in numpy into open3d's
default camera frame (x right, y up, camera on +z looking along -z) instead of moving the camera.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from matplotlib import patches as mpatches

ROOT = Path(__file__).resolve().parent.parent
MARGIN = 0.35  # score.py GOAL_MARGIN_M
VOCAB = ["cabinet", "refrigerator", "shelf", "stove", "bed", "sink", "washer", "toilet", "bathtub", "oven",
         "dishwasher", "fireplace", "stool", "chair", "table", "tv_monitor", "sofa"]
_TAB = {"cabinet": 0, "chair": 2, "table": 4, "sofa": 8, "stool": 12, "refrigerator": 6, "stove": 16, "sink": 18,
        "oven": 10, "fireplace": 3, "dishwasher": 9, "bed": 5, "washer": 1, "toilet": 7, "bathtub": 11, "tv_monitor": 13,
        "shelf": 17}  # tab20 indices; no greys, so nothing blends into the faded point cloud
COLORS = {l: np.array(plt.get_cmap("tab20")(_TAB[l])[:3]) for l in VOCAB}
PLY = {"scene_a": "scene_a_fixed.ply", "scene_b": "scene_b.ply", "scene_c": "scene_c_fixed.ply", "scene_d": "scene_d.ply"}


def in_box(p, box, margin):
    local = (np.asarray(p) - np.asarray(box["center"])) @ np.asarray(box["axes"]).T
    return bool(np.all(np.abs(local) <= np.asarray(box["size"]) / 2 + margin))


def world_up(scene):
    calib = json.loads((ROOT / "data" / scene / "calib.json").read_text())
    R = np.array([f["T_world_camera"] for f in calib["frames"]])[:, :3, :3]
    up = -R[:, :, 1].mean(axis=0)  # camera y points down
    return up / np.linalg.norm(up)


def read_ply(path):
    """Our own binary ply (float xyz + uchar rgb, little endian), as written by scripts/merge_pointcloud.py."""
    raw = Path(path).read_bytes()
    head, body = raw.split(b"end_header\n", 1)
    n = int([l for l in head.decode().splitlines() if l.startswith("element vertex")][0].split()[-1])
    rec = np.frombuffer(body, dtype=[("p", "<f4", 3), ("c", "u1", 3)], count=n)
    return rec["p"].astype(np.float64), rec["c"].astype(np.float64) / 255


# ---- numpy mesh builders (open3d is only used to display them) ------------------------------------------------------


def sphere_mesh(center, r, n_lat=8, n_lon=14):
    lat = np.linspace(0, np.pi, n_lat + 1)[:, None]
    lon = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)[None, :]
    V = np.stack([np.sin(lat) * np.cos(lon), np.sin(lat) * np.sin(lon), np.cos(lat) * np.ones_like(lon)], -1).reshape(-1, 3)
    F = []
    for i in range(n_lat):
        for j in range(n_lon):
            a, b = i * n_lon + j, i * n_lon + (j + 1) % n_lon
            c, d = (i + 1) * n_lon + j, (i + 1) * n_lon + (j + 1) % n_lon
            F += [[a, c, b], [b, c, d]]
    return V * r + np.asarray(center, float), np.array(F, dtype=np.int32)


def cylinder_mesh(a, b, r, n=6):
    d = (b - a) / np.linalg.norm(b - a)
    helper = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(d, helper)
    u /= np.linalg.norm(u)
    v = np.cross(d, u)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.outer(np.cos(ang), u) + np.outer(np.sin(ang), v)
    V = np.vstack([a + r * ring, b + r * ring])
    F = []
    for i in range(n):
        j = (i + 1) % n
        F += [[i, j, n + i], [j, n + j, n + i]]
    return V, np.array(F, dtype=np.int32)


def box_mesh(box, r=0.025):
    c, size, axes = np.array(box["center"]), np.array(box["size"]), np.array(box["axes"])
    signs = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3)
    corners = np.array([c + sum(s[i] * size[i] / 2 * axes[i] for i in range(3)) for s in signs])
    parts = [cylinder_mesh(corners[i], corners[j], r) for i in range(8) for j in range(i + 1, 8) if bin(i ^ j).count("1") == 1]
    return merge(parts)


def merge(parts):
    V, F, off = [], [], 0
    for v, f in parts:
        V.append(v)
        F.append(f + off)
        off += len(v)
    return np.vstack(V), np.vstack(F)


def to_o3d_mesh(V, F, colors):
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.ascontiguousarray(V)), o3d.utility.Vector3iVector(np.ascontiguousarray(F)))
    m.vertex_colors = o3d.utility.Vector3dVector(np.ascontiguousarray(colors))
    return m


# ---- scene ----------------------------------------------------------------------------------------------------------


def room_axes(boxes, up):
    """(lookat, e1, e2): centre of the surveyed boxes and the room's long / short horizontal axes."""
    ctr = np.array([b["center"] for b in boxes])
    lookat = ctr.mean(0)
    hor = ctr - lookat
    hor = hor - np.outer(hor @ up, up)
    e1 = np.linalg.svd(hor, full_matrices=False)[2][0]
    e1 = e1 - (e1 @ up) * up
    e1 /= np.linalg.norm(e1)
    return lookat, e1, np.cross(up, e1)


def load(scene, instances, ceiling, crop, cloud=True):
    boxes = json.loads((ROOT / "data" / scene / "boxes.json").read_text())
    up = world_up(scene)
    lookat, e1, e2 = room_axes(boxes, up)
    ctr = np.array([b["center"] for b in boxes]) - lookat
    lim = [(ctr @ e).min() - crop for e in (e1, e2)], [(ctr @ e).max() + crop for e in (e1, e2)]
    inside = lambda P: np.all([((P - lookat) @ e >= lim[0][k]) & ((P - lookat) @ e <= lim[1][k]) for k, e in enumerate((e1, e2))], axis=0)
    if cloud:
        pts, rgb = read_ply(ROOT / "deliverables/plots" / PLY[scene])
        h = pts @ up
        keep = (h < np.percentile(h, 1) + ceiling) & inside(pts)
        n_out = sum(not inside(np.array(i["center_world"])[None])[0] for i in instances)
        instances = [i for i in instances if inside(np.array(i["center_world"])[None])[0]]
        gray = rgb[keep] @ np.array([0.3, 0.59, 0.11])
        pts, pcol = pts[keep], np.repeat((0.55 + 0.35 * (gray - 0.5))[:, None], 3, axis=1)
    else:  # objects only: nothing to crop against, so every estimate is shown
        pts, pcol, n_out = np.zeros((0, 3)), np.zeros((0, 3)), 0

    gt_parts, gt_cols = [], []
    for b in boxes:
        V, F = box_mesh(b)
        gt_parts.append((V, F))
        gt_cols.append(np.tile(COLORS[b["label"]], (len(V), 1)))
    est_parts, est_cols, n_match = [], [], 0
    for inst in instances:
        ok = any(b["label"] == inst["label"] and in_box(inst["center_world"], b, MARGIN) for b in boxes)
        n_match += ok
        V, F = sphere_mesh(inst["center_world"], 0.16 if ok else 0.07)
        est_parts.append((V, F))
        est_cols.append(np.tile(COLORS[inst["label"]], (len(V), 1)))
    gt = merge(gt_parts) + (np.vstack(gt_cols),)
    est = merge(est_parts) + (np.vstack(est_cols),)
    return dict(boxes=boxes, up=up, pts=pts, pcol=pcol, gt=gt, est=est, n_match=n_match, n_shown=len(instances), n_out=n_out)


def view_frame(boxes, up, name):
    """(lookat, 3x3 rows = camera x, y, z axes) for 'oblique' or 'top-down', aligned with the room's long horizontal axis."""
    lookat, e1, e2 = room_axes(boxes, up)
    front, vup = (0.62 * up - 0.75 * e2 + 0.2 * e1, up) if name == "oblique" else (up, e2)
    z = front / np.linalg.norm(front)
    y = vup - (vup @ z) * z
    y /= np.linalg.norm(y)
    return lookat, np.stack([np.cross(y, z), y, z])


def geometries(d, lookat, Rv):
    tf = lambda P: (P - lookat) @ Rv.T
    (Vg, Fg, Cg), (Ve, Fe, Ce) = d["gt"], d["est"]
    geoms = [to_o3d_mesh(tf(Vg), Fg, Cg), to_o3d_mesh(tf(Ve), Fe, Ce)]
    if len(d["pts"]):
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.ascontiguousarray(tf(d["pts"]))))
        pcd.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(d["pcol"]))
        geoms.insert(0, pcd)
    return geoms


def trim(img, pad=20):
    """Crop the white margin around the rendered scene so it fills its panel."""
    ys, xs = np.nonzero((img[..., :3] < 0.98).any(-1))
    return img[max(ys.min() - pad, 0): ys.max() + pad, max(xs.min() - pad, 0): xs.max() + pad]


def render(geoms, path, zoom, size=(1500, 1100)):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=size[0], height=size[1], visible=False)
    for g in geoms:
        vis.add_geometry(g)
    vis.get_render_option().point_size = 2.0
    vis.reset_view_point(True)
    vis.get_view_control().set_zoom(zoom)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(str(path), do_render=True)
    vis.destroy_window()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*", default=["scene_a", "scene_b"])
    ap.add_argument("--instances", default=str(ROOT / "instances.json"))
    ap.add_argument("--ceiling", type=float, default=2.0, help="drop points more than this many metres above the floor")
    ap.add_argument("--crop", type=float, default=2.0, help="keep scene content within this many metres of the surveyed boxes")
    ap.add_argument("--zoom", type=float, default=0.8)
    ap.add_argument("--no-cloud", action="store_true", help="draw only the surveyed boxes and estimated instances (writes map_<scene>_objects.png)")
    ap.add_argument("--show", action="store_true", help="open an interactive window (oblique start view) instead of writing PNGs")
    args = ap.parse_args()

    allinst = json.loads(Path(args.instances).read_text())
    for scene in args.scenes:
        d = load(scene, allinst[scene], args.ceiling, args.crop, cloud=not args.no_cloud)
        print(f"{scene}: {len(d['boxes'])} surveyed boxes, {len(allinst[scene])} estimated instances; shown {d['n_shown']} ({d['n_match']} match a box), {d['n_out']} outside the crop")
        if args.show:
            lookat, Rv = view_frame(d["boxes"], d["up"], "oblique")
            o3d.visualization.draw_geometries(geometries(d, lookat, Rv), window_name=scene)
            continue
        tmp = {}
        for name in ("oblique", "top-down"):
            lookat, Rv = view_frame(d["boxes"], d["up"], name)
            tmp[name] = ROOT / "deliverables/plots" / f"_{scene}_{name}.png"
            render(geometries(d, lookat, Rv), tmp[name], args.zoom)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6.4))
        for ax, name in zip(axes, tmp):
            ax.imshow(trim(mpimg.imread(tmp[name])))
            ax.set_title(name, fontsize=10)
            ax.axis("off")
            tmp[name].unlink()
        present = [l for l in VOCAB if any(b["label"] == l for b in d["boxes"]) or any(i["label"] == l for i in allinst[scene])]
        handles = [mpatches.Patch(color=COLORS[l], label=l) for l in present]
        handles += [plt.Line2D([], [], color="k", lw=1.5, label="wireframe = surveyed box"),
                    plt.Line2D([], [], marker="o", color="k", ls="", ms=9, label="big sphere = estimate matching a box"),
                    plt.Line2D([], [], marker="o", color="k", ls="", ms=4, label="small sphere = estimate with no matching box")]
        fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8, frameon=False)
        extra = "objects only" if args.no_cloud else f"{d['n_out']} estimates outside the crop not shown"
        fig.suptitle(f"{scene}: surveyed objects vs estimated instances (colour = label); {extra}", fontsize=11)
        fig.tight_layout(rect=[0, 0.2, 1, 0.95])
        out = ROOT / "deliverables/plots" / f"map_{scene}{'_objects' if args.no_cloud else ''}.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    main()
