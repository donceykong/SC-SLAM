#!/usr/bin/env python3
"""
LOAM-style plane (surface) and corner (edge) feature extraction from LiDAR scans.

Ported from DCL-LIO-SAM/src/featureExtraction.cpp and imageProjection.cpp.
Feature extraction is done **per-scan** in the sensor frame (not on a local map).

Algorithm overview:
  1. Organize raw points into a range image (N_SCAN rings × Horizon_SCAN columns)
     using elevation angle for ring assignment and azimuth for column assignment.
  2. Compute range-based curvature: (sum_5_neighbors_each_side − 10·center_range)²
  3. Mark occluded points (depth discontinuities) and parallel-beam points.
  4. Per ring, divide into 6 sectors. In each sector:
     - Extract up to 20 corner features (highest curvature > edge_threshold)
     - Extract all surface features (curvature < surf_threshold)
     - Suppress 5 neighbors on each side of each picked feature.
  5. Voxel-downsample surface features per ring.

Usage:
    python3 extract_features.py --sample-scan 0          # Single scan, sensor frame
    python3 extract_features.py                           # All keyframes, accumulated map
    python3 extract_features.py --map-voxel-size 0.3      # Accumulated map, voxelized

Requires: numpy, scipy, pyvista, tqdm
"""

import argparse
import os

import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from utils import (
    _find_poses_csv, load_poses, apply_imu_to_lidar, get_keyframe_indices,
)

# ── Default LOAM parameters (from DCL-LIO-SAM utility.h / featureExtraction.cpp)

N_SECTORS = 6               # Sectors per ring for feature extraction
MAX_CORNERS_PER_SECTOR = 20  # Max corner features picked per sector
NEIGHBOR_SUPPRESS = 5        # Suppress this many neighbors on each side
COL_DIFF_THRESH = 10         # Column gap threshold for neighbor suppression
OCCLUSION_DEPTH_DIFF = 0.3   # Depth discontinuity for occlusion marking (m)
PARALLEL_BEAM_RATIO = 0.02   # Parallel beam threshold (fraction of range)


# ── Range image organization ──────────────────────────────────────────────────

def organize_scan(pts, n_scan, horizon_scan, min_range, max_range):
    """Organize raw points into scan-line order via a range image.

    Mirrors imageProjection.cpp: assigns ring by elevation angle, column by
    ``atan2(x, y)`` (matching the C++ convention), then extracts valid points
    in row-major (ring, column) order.

    Returns
    -------
    org_pts : (M, 3)  ordered points
    ranges  : (M,)    range per point
    col_ind : (M,)    column index per point
    ring_start, ring_end : (n_scan,)  start/end indices per ring
    """
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    ranges_all = np.sqrt(x**2 + y**2 + z**2)

    # Range filter (lidarMinRange / lidarMaxRange)
    valid = (ranges_all >= min_range) & (ranges_all <= max_range)
    pts = pts[valid]
    ranges_all = ranges_all[valid]
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    # Ring assignment from elevation angle
    xy_dist = np.sqrt(x**2 + y**2)
    elevation = np.arctan2(z, xy_dist)
    elev_min, elev_max = elevation.min(), elevation.max()
    if elev_max - elev_min < 1e-6:
        ring = np.zeros(len(pts), dtype=np.int32)
    else:
        ring = np.clip(
            ((elevation - elev_min) / (elev_max - elev_min) * (n_scan - 1) + 0.5).astype(np.int32),
            0, n_scan - 1,
        )

    # Column assignment – matches C++: atan2(x, y) convention
    horizon_angle_deg = np.arctan2(x, y) * 180.0 / np.pi
    ang_res = 360.0 / horizon_scan
    col = (-np.round((horizon_angle_deg - 90.0) / ang_res) + horizon_scan / 2).astype(np.int32)
    col[col >= horizon_scan] -= horizon_scan
    col[col < 0] += horizon_scan
    col = np.clip(col, 0, horizon_scan - 1)

    # Build range image and extract in row-major order (like cloudExtraction)
    range_img = np.full((n_scan, horizon_scan), np.inf, dtype=np.float32)
    pt_img = np.full((n_scan, horizon_scan, 3), 0.0, dtype=np.float64)

    # First-write wins (matching the C++ "if rangeMat != FLT_MAX: continue")
    for i in range(len(pts)):
        r, c = ring[i], col[i]
        if range_img[r, c] == np.inf:
            range_img[r, c] = ranges_all[i]
            pt_img[r, c] = pts[i]

    # Extract valid points row-major, recording ring start/end.
    # C++ cloudExtraction() insets ring boundaries by +5/−5 so that curvature's
    # 10-point window never spans two rings.  Exact C++ formulas:
    #   startRingIndex[i] = count_before − 1 + 5   (= first_in_ring + 4)
    #   endRingIndex[i]   = count_after  − 1 − 5   (= last_in_ring  − 5)
    org_pts_list = []
    org_ranges_list = []
    org_col_list = []
    ring_start = np.zeros(n_scan, dtype=np.int32)
    ring_end = np.zeros(n_scan, dtype=np.int32)
    count = 0

    for r in range(n_scan):
        first_in_ring = count
        for c in range(horizon_scan):
            if range_img[r, c] != np.inf:
                org_pts_list.append(pt_img[r, c])
                org_ranges_list.append(range_img[r, c])
                org_col_list.append(c)
                count += 1
        n_in_ring = count - first_in_ring
        if n_in_ring > 10:
            ring_start[r] = first_in_ring + 4   # C++: count_before − 1 + 5
            ring_end[r] = count - 1 - 5          # C++: count_after  − 1 − 5
        else:
            # Too few points → sp >= ep → ring will be skipped
            ring_start[r] = first_in_ring
            ring_end[r] = first_in_ring

    if count == 0:
        return np.empty((0, 3)), np.empty(0), np.empty(0, dtype=np.int32), ring_start, ring_end

    org_pts = np.array(org_pts_list, dtype=np.float64)
    ranges = np.array(org_ranges_list, dtype=np.float32)
    col_ind = np.array(org_col_list, dtype=np.int32)

    return org_pts, ranges, col_ind, ring_start, ring_end


# ── Smoothness / curvature ────────────────────────────────────────────────────

def calculate_smoothness(ranges):
    """Compute LOAM curvature for each point (vectorized 10-point window)."""
    n = len(ranges)
    curvature = np.zeros(n, dtype=np.float32)
    if n <= 10:
        return curvature
    # diff = sum of 5 neighbors on each side − 10 × center
    neighbor_sum = np.zeros(n, dtype=np.float64)
    for offset in range(-5, 6):
        if offset == 0:
            continue
        lo = max(0, offset)
        hi = min(n, n + offset)
        src_lo = max(0, -offset)
        src_hi = src_lo + (hi - lo)
        neighbor_sum[lo:hi] += ranges[src_lo:src_hi]
    diff = neighbor_sum - 10.0 * ranges.astype(np.float64)
    curvature[5:n - 5] = (diff[5:n - 5] ** 2).astype(np.float32)
    return curvature


# ── Occlusion / parallel beam marking ────────────────────────────────────────

def mark_occluded_points(ranges, col_ind, n_pts):
    """Mark occluded and parallel-beam points (matching featureExtraction.cpp)."""
    picked = np.zeros(n_pts, dtype=np.int8)

    for i in range(5, n_pts - 6):
        d1 = ranges[i]
        d2 = ranges[i + 1]
        cd = abs(int(col_ind[i + 1]) - int(col_ind[i]))

        if cd < COL_DIFF_THRESH:
            if d1 - d2 > OCCLUSION_DEPTH_DIFF:
                picked[max(0, i - 5):i + 1] = 1
            elif d2 - d1 > OCCLUSION_DEPTH_DIFF:
                picked[i + 1:min(n_pts, i + 7)] = 1

        # Parallel beam
        diff1 = abs(float(ranges[i - 1]) - float(ranges[i]))
        diff2 = abs(float(ranges[i + 1]) - float(ranges[i]))
        if diff1 > PARALLEL_BEAM_RATIO * ranges[i] and diff2 > PARALLEL_BEAM_RATIO * ranges[i]:
            picked[i] = 1

    return picked


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(pts, ranges, col_ind, ring_start, ring_end,
                     n_scan, edge_thresh, surf_thresh, surf_leaf):
    """Extract corner and surface features per ring / sector.

    Returns corner_mask, surface_mask (boolean arrays over *pts*), and curvature.
    """
    n_pts = len(pts)
    curvature = calculate_smoothness(ranges)
    picked = mark_occluded_points(ranges, col_ind, n_pts)
    labels = np.zeros(n_pts, dtype=np.int8)  # 1=corner, -1=surface

    corner_idx = []
    surface_idx = []

    for ring in range(n_scan):
        sp_ring = ring_start[ring]
        ep_ring = ring_end[ring]
        if sp_ring >= ep_ring:
            continue

        ring_surf = []

        for sector in range(N_SECTORS):
            sp = int((sp_ring * (N_SECTORS - sector) + ep_ring * sector) / N_SECTORS)
            ep = int((sp_ring * (N_SECTORS - 1 - sector) + ep_ring * (sector + 1)) / N_SECTORS) - 1
            if sp >= ep:
                continue

            # Sort sector by curvature (ascending)
            idxs = np.arange(sp, ep + 1)
            order = np.argsort(curvature[idxs])
            sorted_idx = idxs[order]

            # ── Corners (highest curvature first) ─────────────────────
            n_picked = 0
            for k in range(len(sorted_idx) - 1, -1, -1):
                ind = sorted_idx[k]
                if picked[ind] == 0 and curvature[ind] > edge_thresh:
                    n_picked += 1
                    if n_picked <= MAX_CORNERS_PER_SECTOR:
                        labels[ind] = 1
                        corner_idx.append(ind)
                    else:
                        break
                    # Suppress neighbors
                    picked[ind] = 1
                    for l in range(1, NEIGHBOR_SUPPRESS + 1):
                        if ind + l >= n_pts:
                            break
                        if abs(int(col_ind[ind + l]) - int(col_ind[ind + l - 1])) > COL_DIFF_THRESH:
                            break
                        picked[ind + l] = 1
                    for l in range(1, NEIGHBOR_SUPPRESS + 1):
                        if ind - l < 0:
                            break
                        if abs(int(col_ind[ind - l]) - int(col_ind[ind - l + 1])) > COL_DIFF_THRESH:
                            break
                        picked[ind - l] = 1

            # ── Surfaces (lowest curvature first) ─────────────────────
            for k in range(len(sorted_idx)):
                ind = sorted_idx[k]
                if picked[ind] == 0 and curvature[ind] < surf_thresh:
                    labels[ind] = -1
                    picked[ind] = 1
                    for l in range(1, NEIGHBOR_SUPPRESS + 1):
                        if ind + l >= n_pts:
                            break
                        if abs(int(col_ind[ind + l]) - int(col_ind[ind + l - 1])) > COL_DIFF_THRESH:
                            break
                        picked[ind + l] = 1
                    for l in range(1, NEIGHBOR_SUPPRESS + 1):
                        if ind - l < 0:
                            break
                        if abs(int(col_ind[ind - l]) - int(col_ind[ind - l + 1])) > COL_DIFF_THRESH:
                            break
                        picked[ind - l] = 1

            # Collect surface candidates (label <= 0) for this sector
            for k in range(sp, ep + 1):
                if labels[k] <= 0:
                    ring_surf.append(k)

        # Voxel-downsample surface features per ring
        if ring_surf and surf_leaf > 0:
            surf_pts = pts[ring_surf]
            keys = np.floor(surf_pts / surf_leaf).astype(np.int64)
            _, ui = np.unique(keys, axis=0, return_index=True)
            surface_idx.extend(ring_surf[i] for i in ui)
        else:
            surface_idx.extend(ring_surf)

    corner_mask = np.zeros(n_pts, dtype=bool)
    surface_mask = np.zeros(n_pts, dtype=bool)
    if corner_idx:
        corner_mask[corner_idx] = True
    if surface_idx:
        surface_mask[surface_idx] = True

    return corner_mask, surface_mask, curvature


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="LOAM-style plane and corner feature extraction from LiDAR scans"
    )
    p.add_argument("--data-dir",
                   default="/media/donceykong/doncey_ssd_02/datasets/CU_MULTI")
    p.add_argument("--env", default="main_campus")
    p.add_argument("--sample-robot", default="robot1")
    p.add_argument("--sample-scan", type=int, default=None,
                   help="If set, process only this keyframe (sensor frame). "
                        "Omit to accumulate all keyframes into a world-frame map.")
    p.add_argument("--keyframe-dist", type=float, default=5.0)
    p.add_argument("--map-voxel-size", type=float, default=0.0,
                   help="Voxel size (m) for final accumulated map downsampling (0=off)")

    # LiDAR configuration
    p.add_argument("--n-scan", type=int, default=16,
                   help="Number of LiDAR rings")
    p.add_argument("--horizon-scan", type=int, default=1800,
                   help="Horizontal resolution")
    p.add_argument("--min-range", type=float, default=1.0,
                   help="Minimum range filter (m)")
    p.add_argument("--max-range", type=float, default=1000.0,
                   help="Maximum range filter (m)")

    # LOAM thresholds
    p.add_argument("--edge-threshold", type=float, default=1.0,
                   help="Min curvature for corner features")
    p.add_argument("--surf-threshold", type=float, default=0.1,
                   help="Max curvature for surface features")
    p.add_argument("--surf-leaf-size", type=float, default=0.1,
                   help="Voxel leaf size for surface downsampling (m)")

    p.add_argument("--point-size", type=float, default=3.0)
    args = p.parse_args()

    # ── Load robot data ───────────────────────────────────────────────────
    robot_dir = os.path.join(args.data_dir, args.env, args.sample_robot)
    poses_csv = _find_poses_csv(robot_dir, args.sample_robot, args.env)
    scan_bin_dir = os.path.join(robot_dir, "lidar_bin", "data")

    if not poses_csv or not os.path.isdir(scan_bin_dir):
        print("ERROR: Could not find poses or scan data"); return

    poses = load_poses(poses_csv)
    apply_imu_to_lidar(poses)
    scan_files = sorted(f for f in os.listdir(scan_bin_dir) if f.endswith(".bin"))
    n_avail = min(len(scan_files), len(poses))

    kf_indices = get_keyframe_indices(poses, n_avail, args.keyframe_dist)

    if args.sample_scan is not None:
        # ── Single scan mode (sensor frame) ───────────────────────────────
        if args.sample_scan >= len(kf_indices):
            print(f"ERROR: sample-scan {args.sample_scan} out of range "
                  f"(max: {len(kf_indices) - 1})"); return

        scan_idx = kf_indices[args.sample_scan]
        path = os.path.join(scan_bin_dir, scan_files[scan_idx])
        raw = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        sensor_pts = raw[:, :3].astype(np.float64)
        print(f"Loaded {scan_files[scan_idx]}: {len(sensor_pts):,} points")

        org_pts, ranges, col_ind, ring_start, ring_end = organize_scan(
            sensor_pts, args.n_scan, args.horizon_scan,
            args.min_range, args.max_range,
        )
        if len(org_pts) == 0:
            print("ERROR: no valid points after range filtering"); return

        corner_mask, surface_mask, _ = extract_features(
            org_pts, ranges, col_ind, ring_start, ring_end,
            args.n_scan, args.edge_threshold, args.surf_threshold,
            args.surf_leaf_size,
        )

        corner_pts = org_pts[corner_mask]
        surface_pts = org_pts[surface_mask]
        other_pts = org_pts[~corner_mask & ~surface_mask]
        title = (f"LOAM Feature Extraction | {args.sample_robot} "
                 f"kf={args.sample_scan} | Corners: {len(corner_pts):,} | "
                 f"Surfaces: {len(surface_pts):,} | sensor frame")

    else:
        # ── Accumulated map mode (world frame) ────────────────────────────
        print(f"Processing {len(kf_indices)} keyframes...")
        all_corners = []
        all_surfaces = []
        all_other = []

        for kf_ix in tqdm(kf_indices, desc="Extracting features"):
            pose = poses[kf_ix]
            rot = Rotation.from_quat(pose[4:8])
            t = pose[1:4]

            path = os.path.join(scan_bin_dir, scan_files[kf_ix])
            raw = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
            sensor_pts = raw[:, :3].astype(np.float64)

            org_pts, ranges, col_ind, ring_start, ring_end = organize_scan(
                sensor_pts, args.n_scan, args.horizon_scan,
                args.min_range, args.max_range,
            )
            if len(org_pts) == 0:
                continue

            corner_mask, surface_mask, _ = extract_features(
                org_pts, ranges, col_ind, ring_start, ring_end,
                args.n_scan, args.edge_threshold, args.surf_threshold,
                args.surf_leaf_size,
            )

            # Transform to world frame and voxel downsample per scan
            world_pts = rot.apply(org_pts) + t
            c_pts = world_pts[corner_mask]
            s_pts = world_pts[surface_mask]
            o_pts = world_pts[~corner_mask & ~surface_mask]

            if args.map_voxel_size > 0:
                v = args.map_voxel_size
                if len(c_pts) > 0:
                    keys = np.floor(c_pts / v).astype(np.int64)
                    _, ui = np.unique(keys, axis=0, return_index=True)
                    c_pts = c_pts[ui]
                if len(s_pts) > 0:
                    keys = np.floor(s_pts / v).astype(np.int64)
                    _, ui = np.unique(keys, axis=0, return_index=True)
                    s_pts = s_pts[ui]
                if len(o_pts) > 0:
                    keys = np.floor(o_pts / v).astype(np.int64)
                    _, ui = np.unique(keys, axis=0, return_index=True)
                    o_pts = o_pts[ui]

            all_corners.append(c_pts)
            all_surfaces.append(s_pts)
            all_other.append(o_pts)

        corner_pts = np.vstack(all_corners) if all_corners else np.empty((0, 3))
        surface_pts = np.vstack(all_surfaces) if all_surfaces else np.empty((0, 3))
        other_pts = np.vstack(all_other) if all_other else np.empty((0, 3))

        print(f"Total: corners {len(corner_pts):,}, "
              f"surfaces {len(surface_pts):,}, other {len(other_pts):,}")
        title = (f"LOAM Feature Map | {args.sample_robot} | "
                 f"{len(kf_indices)} keyframes | "
                 f"Corners: {len(corner_pts):,} | Surfaces: {len(surface_pts):,}")

    # ── Visualize ─────────────────────────────────────────────────────────
    plotter = pv.Plotter()

    if len(other_pts) > 0:
        cloud = pv.PolyData(other_pts)
        plotter.add_mesh(cloud, color=[0.5, 0.5, 0.5],
                         point_size=args.point_size * 0.5,
                         render_points_as_spheres=False,
                         opacity=0.3,
                         label=f"Other ({len(other_pts):,})")

    # if len(surface_pts) > 0:
    #     cloud = pv.PolyData(surface_pts)
    #     plotter.add_mesh(cloud, color=[0.2, 0.6, 1.0],
    #                      point_size=args.point_size,
    #                      render_points_as_spheres=False,
    #                      label=f"Surface ({len(surface_pts):,})")

    if len(corner_pts) > 0:
        cloud = pv.PolyData(corner_pts)
        plotter.add_mesh(cloud, color=[1.0, 0.2, 0.2],
                         point_size=args.point_size * 1.5,
                         render_points_as_spheres=False,
                         label=f"Corner ({len(corner_pts):,})")

    plotter.add_legend(bcolor=(0.9, 0.9, 0.9), face=None)
    plotter.set_background("white")
    plotter.add_text(title, font_size=10, color="black")

    print("\nLaunching PyVista viewer...")
    plotter.show()


if __name__ == "__main__":
    main()
