"""Pinhole projection and rigid-transform helpers. Camera frame is OpenCV: x right, y down, z forward."""

import numpy as np


def intrinsic_matrix(cam):
    return np.array(
        [[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1]],
        dtype=np.float64,
    )


def unproject(depth_m, cam, stride=1):
    """Depth image (metres, 0 = invalid) -> (N,3) camera-frame points and their (N,2) pixel coords."""
    v, u = np.mgrid[0 : depth_m.shape[0] : stride, 0 : depth_m.shape[1] : stride]
    z = depth_m[v, u]
    ok = z > 0
    u, v, z = (
        u[ok].astype(np.float64),
        v[ok].astype(np.float64),
        z[ok].astype(np.float64),
    )
    x = (u - cam["cx"]) * z / cam["fx"]
    y = (v - cam["cy"]) * z / cam["fy"]
    return np.stack([x, y, z], axis=1), np.stack([u, v], axis=1)


def project(points_cam, cam):
    """(N,3) camera-frame points -> (N,2) pixel coords and (N,) depths; points behind the camera get depth <= 0."""
    z = points_cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = cam["fx"] * points_cam[:, 0] / z + cam["cx"]
        v = cam["fy"] * points_cam[:, 1] / z + cam["cy"]
    return np.stack([u, v], axis=1), z


def transform(T, points):
    return points @ T[:3, :3].T + T[:3, 3]


def rotation_from_axis_angle(rvec):
    rvec = np.asarray(rvec, dtype=np.float64)
    theta = np.linalg.norm(rvec)
    if theta < 1e-12:
        return np.eye(3)
    k = rvec / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


def rotation_about_axis(axis, angle_rad):
    axis = np.asarray(axis, dtype=np.float64)
    return rotation_from_axis_angle(axis / np.linalg.norm(axis) * angle_rad)


def yaw_between(T_a, T_b):
    """Angle in radians between the optical axes of two camera poses."""
    za, zb = T_a[:3, 2], T_b[:3, 2]
    return float(np.arccos(np.clip(np.dot(za, zb), -1.0, 1.0)))


def aligned_depth_to_color(depth_raw, depth_cam, color_cam, T_color_depth, scale_m=None):
    """Warp a raw depth image into the colour camera: HxW float32 metres (z along the colour optical axis, 0 = no return).

    raw units -> metres with depth_cam["scale_m"] (or scale_m if given, e.g. a corrected value), unproject with the
    depth intrinsics, move to the colour frame with T_color_depth (depth sensor -> colour camera), project with the
    colour intrinsics. Where several depth pixels land on one colour pixel the nearest wins (z-buffer). This is a
    forward warp, so colour pixels no depth pixel maps to stay 0: expect holes when the two cameras differ in
    resolution/focal length and an empty border strip when the principal points differ.
    """
    scale = depth_cam["scale_m"] if scale_m is None else scale_m
    pts, _ = unproject(depth_raw.astype(np.float32) * scale, depth_cam)
    uv, z = project(transform(T_color_depth, pts), color_cam)
    u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
    h, w = color_cam["height"], color_cam["width"]
    ok = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    out = np.full((h, w), np.inf, dtype=np.float32)
    np.minimum.at(out, (v[ok], u[ok]), z[ok].astype(np.float32))
    out[np.isinf(out)] = 0
    return out
