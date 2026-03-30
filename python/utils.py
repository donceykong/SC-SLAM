#!/usr/bin/env python3
"""
Shared utilities for CU_MULTI lidar processing scripts.
"""

import os

import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation

# ── Platform constants ────────────────────────────────────────────────────────

IMU_TO_LIDAR_T = np.array([-0.058038, 0.015573, 0.049603])
IMU_TO_LIDAR_Q = [0.0, 0.0, 1.0, 0.0]  # [qx, qy, qz, qw]

# ── CRS / LAS constants ───────────────────────────────────────────────────────

FT_TO_M = 0.3048006096

_WGS84_P4 = "+proj=longlat +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +no_defs"
_UTM13N = "EPSG:32613"

CO_STATE_PLANE_PROJ4 = {
    "colorado north": (
        "+proj=lcc +lat_1=39.71666666666667 +lat_2=40.78333333333333 "
        "+lat_0=39.33333333333334 +lon_0=-105.5 "
        "+x_0=914401.8289 +y_0=304800.6096 "
        "+ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=us-ft +no_defs"
    ),
    "colorado central": (
        "+proj=lcc +lat_1=38.45 +lat_2=39.75 "
        "+lat_0=37.83333333333334 +lon_0=-105.5 "
        "+x_0=914401.8289 +y_0=304800.6096 "
        "+ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=us-ft +no_defs"
    ),
}

# ── Pose loading ──────────────────────────────────────────────────────────────

def _find_poses_csv(robot_dir: str, robot_name: str, env_name: str):
    """Locate the poses CSV inside *robot_dir*."""
    expected = os.path.join(robot_dir, f"{robot_name}_{env_name}_gt_utm_poses.csv")
    if os.path.isfile(expected):
        return expected
    for f in os.listdir(robot_dir):
        if f.endswith("_utm_poses.csv"):
            return os.path.join(robot_dir, f)
    fb = os.path.join(robot_dir, "poses.csv")
    return fb if os.path.isfile(fb) else None


def load_poses(csv_path: str) -> np.ndarray:
    """Read poses CSV → (N, 8+) array [timestamp, x, y, z, qx, qy, qz, qw]."""
    rows = []
    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 8:
                continue
            try:
                rows.append([float(v) for v in parts])
            except ValueError:
                continue
    return np.array(rows)


def apply_imu_to_lidar(poses: np.ndarray) -> None:
    """Transform IMU-frame poses to LiDAR-frame in-place."""
    rot_ext = Rotation.from_quat(IMU_TO_LIDAR_Q).as_matrix()
    for i in range(len(poses)):
        R_imu = Rotation.from_quat(poses[i, 4:8]).as_matrix()
        poses[i, 1:4] += R_imu @ IMU_TO_LIDAR_T
        poses[i, 4:8] = Rotation.from_matrix(R_imu @ rot_ext).as_quat()


def get_keyframe_indices(poses: np.ndarray, n_avail: int, keyframe_dist: float = 20.0):
    """Select keyframe indices by minimum distance between consecutive poses."""
    indices = [0]
    last_pos = poses[0, 1:4].copy()
    for i in range(1, n_avail):
        pos = poses[i, 1:4]
        if np.linalg.norm(pos - last_pos) >= keyframe_dist:
            indices.append(i)
            last_pos = pos.copy()
    return indices

# ── CRS helpers ───────────────────────────────────────────────────────────────

def detect_feet(crs_obj) -> bool:
    """Return True if *crs_obj* uses US survey feet."""
    try:
        unit = crs_obj.axis_info[0].unit_name.lower()
        if any(k in unit for k in ("foot", "feet", "ft")):
            return True
    except Exception:
        pass
    s = (str(crs_obj) + " " + crs_obj.to_proj4()).lower()
    return any(k in s for k in ("us-ft", "ftus", "foot", "feet"))


def resolve_crs(las_files, crs_arg=None):
    """Determine the native CRS of *las_files* (first file used). Requires pyproj."""
    from pyproj import CRS, Transformer  # lazy: only scripts using LAS need pyproj
    las0 = las_files[0]
    mid_x = (las0.header.x_min + las0.header.x_max) / 2
    mid_y = (las0.header.y_min + las0.header.y_max) / 2

    def _valid(crs_obj):
        try:
            to_wgs = Transformer.from_crs(crs_obj, _WGS84_P4, always_xy=True)
            lon, lat = to_wgs.transform(mid_x, mid_y)
            return np.isfinite(lon) and np.isfinite(lat) and abs(lat) < 90
        except Exception:
            return False

    if crs_arg:
        try:
            c = CRS.from_user_input(crs_arg)
            if _valid(c):
                return c
        except Exception:
            pass

    for key, proj4 in CO_STATE_PLANE_PROJ4.items():
        try:
            c = CRS.from_proj4(proj4)
            if _valid(c):
                print(f"  Auto-detected CRS '{key}'")
                return c
        except Exception:
            pass
    return None

# ── Z-only ICP ────────────────────────────────────────────────────────────────

def icp_z_only(source, target, max_iterations=50, tolerance=1e-4,
               max_correspondence_dist=5.0):
    """Compute a pure vertical shift aligning *source* to *target*."""
    tree = KDTree(target[:, :2])
    dz_total = 0.0
    for _ in range(max_iterations):
        shifted = source.copy()
        shifted[:, 2] += dz_total
        d2, idx = tree.query(shifted[:, :2])
        ok = d2 < max_correspondence_dist
        if ok.sum() < 10:
            break
        dz = target[idx[ok], 2] - shifted[ok, 2]
        med = np.median(dz)
        std = np.std(dz)
        if std > 0:
            keep = np.abs(dz - med) < 2 * std
            if keep.sum() > 10:
                dz = dz[keep]
        corr = np.median(dz)
        dz_total += corr
        if abs(corr) < tolerance:
            break
    shifted = source.copy()
    shifted[:, 2] += dz_total
    d2, idx = tree.query(shifted[:, :2])
    ok = d2 < max_correspondence_dist
    err = np.mean(np.abs(target[idx[ok], 2] - shifted[ok, 2])) if ok.sum() else float("inf")
    return dz_total, err, int(ok.sum())
