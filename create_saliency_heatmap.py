#!/usr/bin/env python3
"""
Process scans from a robot, find other robot scans within a pose distance,
compute overlap heatmaps, and optionally visualize in PyVista.

Processes all keyframe scans from the sample robot by default. Use --sample-scan N
to process only one scan. Use --visualize to show PyVista (blocks until window
closed; omit for batch processing).

Usage:
    python3 testing.py                    # Process all robot1 scans, no viz
    python3 testing.py --visualize        # Process all, show last scan at end
    python3 testing.py --sample-scan 0    # Process only keyframe 0
    python3 testing.py --sample-scan 0 --visualize  # Process one, show viz

    # Example: process keyframe 50, show heatmap, show progress, 
    # and visualize the result
    
    python3 testing.py --sample-scan 50 --heatmap \
        --min-dist 10 --max-dist 100 --n-dist 0.1 \
        --voxel-leaf 0.1 --show-peer-progress --visualize

    python3 testing.py --multiprocess --cores 4  # Parallel processing

Overlap weight: farther orientation counts more (weight = orient_angle).

Unlabeled (0), tree (10), and vehicle (11) points are excluded from overlap counting
when groundtruth_labels/ are present (from overlay_scans_sequential --save-labels).

Requires: pyvista, numpy, scipy, tqdm. Optional: open3d (for --fpfh-weight).
"""

import argparse
import os
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm
import pyvista as pv
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation

# IMU → LiDAR extrinsic (CU_MULTI platform)
IMU_TO_LIDAR_T = np.array([-0.058038, 0.015573, 0.049603])
IMU_TO_LIDAR_Q = [0.0, 0.0, 1.0, 0.0]  # [qx, qy, qz, qw]

# Labels to ignore in overlap counting (per build_map_from_labels / view_DSM_DEM_OSM_SCANS)
LABEL_IGNORE = {0, 10, 11}  # unlabeled, vehicle, tree


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


def filter_distance(
    pts: np.ndarray, center: np.ndarray, min_dist: float, max_dist: float
) -> np.ndarray:
    """Filter points to those within [min_dist, max_dist] of center."""
    if min_dist is None and max_dist is None:
        return pts
    d = np.linalg.norm(pts - center, axis=1)
    mask = np.ones(len(pts), dtype=bool)
    if min_dist is not None:
        mask &= d >= min_dist
    if max_dist is not None:
        mask &= d <= max_dist
    return pts[mask]


def voxel_downsample(pts: np.ndarray, leaf_size: float) -> np.ndarray:
    """Downsample points by voxel grid (keep one point per voxel)."""
    if leaf_size <= 0 or len(pts) == 0:
        return pts
    keys = np.floor(pts / leaf_size).astype(np.int64)
    _, ui = np.unique(keys, axis=0, return_index=True)
    return pts[ui]


def load_labels_for_scan(robot_dir: str, idx: int, n_expected: int):
    """Load labels from groundtruth_labels/{idx:010d}.bin. Returns None if missing or length mismatch."""
    label_path = os.path.join(robot_dir, "groundtruth_labels", f"{idx:010d}.bin")
    if not os.path.isfile(label_path):
        return None
    lbl = np.fromfile(label_path, dtype=np.uint32)
    if len(lbl) != n_expected:
        return None
    return lbl


def load_scan_at_index(
    scan_bin_dir: str, scan_files: list, poses: np.ndarray, idx: int
) -> np.ndarray:
    """Load scan at pose index and transform to world frame (UTM)."""
    if idx >= len(scan_files) or idx >= len(poses):
        raise IndexError(f"Scan index {idx} out of range")
    pose = poses[idx]
    tx, ty, tz = pose[1], pose[2], pose[3]
    rot = Rotation.from_quat(pose[4:8])
    path = os.path.join(scan_bin_dir, scan_files[idx])
    raw = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    pts = rot.apply(raw[:, :3].astype(np.float64)) + np.array([tx, ty, tz])
    return pts


def get_robot_scan_data(data_dir: str, env: str, robot_name: str):
    """Load poses and scan metadata for a robot. Returns (poses, scan_files, scan_bin_dir) or None."""
    robot_dir = os.path.join(data_dir, env, robot_name)
    poses_csv = _find_poses_csv(robot_dir, robot_name, env)
    scan_bin_dir = os.path.join(robot_dir, "lidar_bin", "data")
    if not poses_csv or not os.path.isdir(scan_bin_dir):
        return None
    poses = load_poses(poses_csv)
    apply_imu_to_lidar(poses)
    scan_files = sorted(f for f in os.listdir(scan_bin_dir) if f.endswith(".bin"))
    n_avail = min(len(scan_files), len(poses))
    return {
        "poses": poses,
        "scan_files": scan_files,
        "scan_bin_dir": scan_bin_dir,
        "n_avail": n_avail,
    }


def quaternion_angle_diff(q1: np.ndarray, q2: np.ndarray) -> float:
    """Angle (radians) between two rotations represented by unit quaternions [qx,qy,qz,qw]."""
    dot = np.abs(np.dot(q1, q2))
    return 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))


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


def _compute_fpfh_descriptiveness(pts: np.ndarray, voxel_leaf: float) -> np.ndarray:
    """Compute per-point descriptiveness from FPFH (0-1). Requires open3d."""
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError("open3d required for --fpfh-weight. Install with: pip install open3d")

    if len(pts) < 10:
        return np.ones(len(pts), dtype=np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))

    radius_normal = 4 #max(voxel_leaf * 1, 0.5)
    radius_feature = 4 #max(voxel_leaf * 3, 2)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=100)
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    fpfh_arr = np.asarray(fpfh.data).T  # (N, 33)

    # Descriptiveness: mean distance to k-NN in FPFH space (distinctive points are farther)
    k = min(10, len(pts) - 1)
    tree = KDTree(fpfh_arr)
    dists, _ = tree.query(fpfh_arr, k=k + 1)  # +1 to exclude self
    mean_dist = np.mean(dists[:, 1:], axis=1)
    lo, hi = mean_dist.min(), mean_dist.max()
    if hi > lo:
        descriptiveness = (mean_dist - lo) / (hi - lo)
    else:
        descriptiveness = np.ones(len(pts), dtype=np.float64) * 0.5
    return descriptiveness.astype(np.float64)


def _process_single_scan(task):
    """Worker: process one sample scan, compute overlap, save heatmap. Returns None."""
    (kf_ix, sample_kf_idx, poses, scan_files, scan_bin_dir, other_robot_data,
     heatmap_dir, distance, min_dist, max_dist, n_dist, orientation_scale,
     voxel_leaf, normalize_mode, fpfh_weight) = task

    sample_pose = poses[sample_kf_idx]
    sample_pos = sample_pose[1:4]
    sample_quat = sample_pose[4:8]
    orient_scale = orientation_scale if orientation_scale is not None else distance / np.pi

    raw_sample_pts = load_scan_at_index(scan_bin_dir, scan_files, poses, sample_kf_idx)
    filtered_sample_pts = filter_distance(raw_sample_pts, sample_pos, min_dist, max_dist)
    sample_pts = voxel_downsample(filtered_sample_pts, voxel_leaf)

    robot_dir = os.path.dirname(os.path.dirname(scan_bin_dir))
    raw_labels = load_labels_for_scan(robot_dir, sample_kf_idx, len(raw_sample_pts))
    if raw_labels is not None:
        d = np.linalg.norm(raw_sample_pts - sample_pos, axis=1)
        mask = np.ones(len(raw_sample_pts), dtype=bool)
        if min_dist is not None:
            mask &= d >= min_dist
        if max_dist is not None:
            mask &= d <= max_dist
        filtered_labels = raw_labels[mask]
        if voxel_leaf > 0:
            keys = np.floor(filtered_sample_pts / voxel_leaf).astype(np.int64)
            _, ui = np.unique(keys, axis=0, return_index=True)
            sample_labels = filtered_labels[ui]
        else:
            sample_labels = filtered_labels
    else:
        sample_labels = None

    to_load = []
    for rname, rdata in other_robot_data.items():
        r_poses = rdata["poses"]
        r_n = rdata["n_avail"]
        r_kf = rdata.get("kf_indices")
        for i in range(r_n):
            if r_kf is not None and i not in r_kf:
                continue
            pos = r_poses[i, 1:4]
            if np.linalg.norm(pos - sample_pos) <= distance:
                to_load.append((rname, rdata, i))

    overlap_sum = np.zeros(len(sample_pts), dtype=np.float64)
    for rname, rdata, i in to_load:
        r_poses = rdata["poses"]
        r_files = rdata["scan_files"]
        r_dir = rdata["scan_bin_dir"]
        pos = r_poses[i, 1:4]
        quat = r_poses[i, 4:8]
        orient_angle = quaternion_angle_diff(sample_quat, quat)
        pts = load_scan_at_index(r_dir, r_files, r_poses, i)
        pts = filter_distance(pts, pos, min_dist, max_dist)
        pts = voxel_downsample(pts, voxel_leaf)

        diff = max(0.01, orient_scale * orient_angle)
        weight = diff  # Farther scans count more
        if len(pts) > 0 and weight > 0:
            tree = KDTree(pts)
            hits = tree.query_ball_point(sample_pts, n_dist)
            for j, hit in enumerate(hits):
                if len(hit) > 0 and (sample_labels is None or sample_labels[j] not in LABEL_IGNORE):
                    overlap_sum[j] += weight

    if fpfh_weight > 0 and len(sample_pts) >= 10:
        desc = _compute_fpfh_descriptiveness(sample_pts, voxel_leaf)
        overlap_sum *= 1.0 + fpfh_weight * desc

    if overlap_sum.max() > 0:
        if normalize_mode == "log":
            vals = np.log1p(overlap_sum)
        elif normalize_mode == "sqrt":
            vals = np.sqrt(overlap_sum)
        else:
            vals = overlap_sum
        normalized_overlap = vals / vals.max()
        sample_scan_name = scan_files[sample_kf_idx]
        heatmap_path = os.path.join(heatmap_dir, sample_scan_name)
        tree = KDTree(sample_pts)
        _, nn_idx = tree.query(raw_sample_pts, k=1)
        nn_idx = np.atleast_1d(nn_idx.ravel())
        scan_heatmap = normalized_overlap[nn_idx].astype(np.float32)
        scan_heatmap.tofile(heatmap_path)

    return kf_ix


def main():
    p = argparse.ArgumentParser(
        description="View sample scan (green) + nearby scans from other robots (red) in PyVista"
    )
    p.add_argument(
        "--data-dir",
        default="/media/donceykong/doncey_ssd_02/datasets/CU_MULTI",
        help="Root dataset directory",
    )
    p.add_argument(
        "--env",
        default="main_campus",
        help="Environment name (sub-folder under data-dir)",
    )
    p.add_argument(
        "--sample-robot",
        default="robot1",
        help="Robot to take the sample scan from",
    )
    p.add_argument(
        "--sample-scan",
        type=int,
        default=None,
        help="If set, process only this keyframe index; else process all keyframes",
    )
    p.add_argument(
        "--distance",
        type=float,
        default=15.0,
        help="Max pose distance (m) to include scans from other robots",
    )
    p.add_argument(
        "--voxel-leaf",
        type=float,
        default=0.1,
        help="Voxel leaf size (m) for downsampling",
    )
    p.add_argument(
        "--min-dist",
        type=float,
        default=None,
        help="Min point distance (m) from robot pose to keep",
    )
    p.add_argument(
        "--max-dist",
        type=float,
        default=None,
        help="Max point distance (m) from robot pose to keep",
    )
    p.add_argument(
        "--n-dist",
        type=float,
        default=0.2,
        help="Neighbor distance (m) for overlap counting (used with --heatmap)",
    )
    p.add_argument(
        "--orientation-scale",
        type=float,
        default=None,
        help="Weight scale for orientation. Default: distance/pi",
    )
    p.add_argument(
        "--heatmap",
        action="store_true",
        help="When visualizing, show overlap heatmap instead of green/red view",
    )
    p.add_argument(
        "--normalize",
        choices=["linear", "log", "sqrt"],
        default="linear",
        help="Overlap heatmap normalization: linear (divide by max), log (log1p), sqrt (sqrt). Default: linear",
    )
    p.add_argument(
        "--fpfh-weight",
        type=float,
        default=0.0,
        help="Boost overlap for geometrically distinctive points (FPFH). 0=off. Requires open3d.",
    )
    p.add_argument(
        "--visualize",
        action="store_true",
        help="Show PyVista viewer (blocks until closed; omit for batch processing)",
    )
    p.add_argument(
        "--keyframe-dist",
        type=float,
        default=5.0,
        help="Min distance (m) between keyframes for sample robot",
    )
    p.add_argument(
        "--peer-keyframe-dist",
        type=float,
        default=0.5,
        help="Min distance (m) between peer robot keyframes. Limits accumulation when a robot remains still. Omit to use all peer scans.",
    )
    p.add_argument(
        "--show-peer-progress",
        action="store_true",
        help="Show tqdm progress for peer scan processing per sample scan",
    )
    p.add_argument(
        "--multiprocess",
        action="store_true",
        help="Process sample scans in parallel (incompatible with --visualize and --show-peer-progress)",
    )
    p.add_argument(
        "--cores",
        type=int,
        default=None,
        help="Number of worker processes for multiprocessing (default: CPU count - 1)",
    )
    args = p.parse_args()

    if args.multiprocess:
        if args.visualize:
            raise SystemExit("ERROR: --multiprocess cannot be used with --visualize")
        if args.show_peer_progress:
            raise SystemExit("ERROR: --multiprocess cannot be used with --show-peer-progress")

    # Other robots to search (main campus robots 2-4)
    other_robots = ["robot2", "robot3", "robot4"]

    # Load sample robot data
    print(f"Loading {args.sample_robot} from {args.env}...")
    sample_data = get_robot_scan_data(args.data_dir, args.env, args.sample_robot)
    if sample_data is None:
        print(f"ERROR: Could not load {args.sample_robot} data")
        return

    poses = sample_data["poses"]
    scan_files = sample_data["scan_files"]
    scan_bin_dir = sample_data["scan_bin_dir"]
    n_avail = sample_data["n_avail"]

    # Keyframe selection for sample robot
    kf_indices = get_keyframe_indices(poses, n_avail, args.keyframe_dist)
    if args.sample_scan is not None:
        if args.sample_scan >= len(kf_indices):
            print(f"ERROR: sample-scan {args.sample_scan} out of range (max: {len(kf_indices) - 1})")
            return
        kf_indices = [kf_indices[args.sample_scan]]

    print(f"  Processing {len(kf_indices)} keyframe scans")

    # Pre-load other robot data (reused across scans)
    other_robot_data = {}
    for rname in other_robots:
        rdata = get_robot_scan_data(args.data_dir, args.env, rname)
        if rdata is not None:
            if args.peer_keyframe_dist is not None:
                kf = set(get_keyframe_indices(rdata["poses"], rdata["n_avail"], args.peer_keyframe_dist))
                rdata["kf_indices"] = kf
            other_robot_data[rname] = rdata
        else:
            print(f"  WARNING: Could not load {rname}")

    if args.peer_keyframe_dist is not None:
        print(f"  Peer keyframe dist: {args.peer_keyframe_dist}m (limiting peer scans per robot)")

    if args.fpfh_weight > 0:
        try:
            import open3d  # noqa: F401
        except ImportError:
            raise SystemExit("ERROR: --fpfh-weight requires open3d. Install with: pip install open3d")

    robot_dir = os.path.join(args.data_dir, args.env, args.sample_robot)
    heatmap_dir = os.path.join(robot_dir, "heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)

    last_viz_data = None  # For --visualize at end

    orient_scale_val = args.orientation_scale if args.orientation_scale is not None else args.distance / np.pi
    if args.multiprocess:
        n_cores = args.cores if args.cores is not None else max(1, (os.cpu_count() or 1) - 1)
        print(f"  Using {n_cores} worker processes")
        tasks = [
            (kf_ix, kf_indices[kf_ix], poses, scan_files, scan_bin_dir, other_robot_data,
             heatmap_dir, args.distance, args.min_dist, args.max_dist, args.n_dist,
             orient_scale_val, args.voxel_leaf, args.normalize, args.fpfh_weight)
            for kf_ix in range(len(kf_indices))
        ]
        with Pool(processes=n_cores) as pool:
            list(tqdm(
                pool.imap_unordered(_process_single_scan, tasks),
                total=len(tasks),
                desc="Processing robot1 scans",
            ))
    else:
        for kf_ix, sample_kf_idx in enumerate(tqdm(kf_indices, desc="Processing robot1 scans")):
            sample_pose = poses[sample_kf_idx]
            sample_pos = sample_pose[1:4]

            # Load sample scan
            raw_sample_pts = load_scan_at_index(
                scan_bin_dir, scan_files, poses, sample_kf_idx
            )
            filtered_sample_pts = filter_distance(
                raw_sample_pts, sample_pos, args.min_dist, args.max_dist
            )
            sample_pts = voxel_downsample(filtered_sample_pts, args.voxel_leaf)

            robot_dir = os.path.join(args.data_dir, args.env, args.sample_robot)
            raw_labels = load_labels_for_scan(robot_dir, sample_kf_idx, len(raw_sample_pts))
            if raw_labels is not None:
                d = np.linalg.norm(raw_sample_pts - sample_pos, axis=1)
                mask = np.ones(len(raw_sample_pts), dtype=bool)
                if args.min_dist is not None:
                    mask &= d >= args.min_dist
                if args.max_dist is not None:
                    mask &= d <= args.max_dist
                filtered_labels = raw_labels[mask]
                if args.voxel_leaf > 0:
                    keys = np.floor(filtered_sample_pts / args.voxel_leaf).astype(np.int64)
                    _, ui = np.unique(keys, axis=0, return_index=True)
                    sample_labels = filtered_labels[ui]
                else:
                    sample_labels = filtered_labels
            else:
                sample_labels = None

            # Find nearby scans from robots 2-4
            to_load = []
            for rname, rdata in other_robot_data.items():
                r_poses = rdata["poses"]
                r_n = rdata["n_avail"]
                r_kf = rdata.get("kf_indices")
                for i in range(r_n):
                    if r_kf is not None and i not in r_kf:
                        continue
                    pos = r_poses[i, 1:4]
                    if np.linalg.norm(pos - sample_pos) <= args.distance:
                        to_load.append((rname, rdata, i))

            sample_quat = sample_pose[4:8]
            orient_scale = args.orientation_scale if args.orientation_scale is not None else args.distance / np.pi
            overlap_sum = np.zeros(len(sample_pts), dtype=np.float64)
            nearby_scans = []  # Only populated when args.visualize
            peer_iter = tqdm(to_load, desc="Peer scans", leave=False) if args.show_peer_progress else to_load
            for rname, rdata, i in peer_iter:
                r_poses = rdata["poses"]
                r_files = rdata["scan_files"]
                r_dir = rdata["scan_bin_dir"]
                pos = r_poses[i, 1:4]
                quat = r_poses[i, 4:8]
                orient_angle = quaternion_angle_diff(sample_quat, quat)
                pts = load_scan_at_index(r_dir, r_files, r_poses, i)
                pts = filter_distance(pts, pos, args.min_dist, args.max_dist)
                pts = voxel_downsample(pts, args.voxel_leaf)

                diff = max(0.01, orient_scale * orient_angle)
                weight = diff  # Farther scans count more
                if len(pts) > 0 and weight > 0:
                    tree = KDTree(pts)
                    hits = tree.query_ball_point(sample_pts, args.n_dist)
                    for j, hit in enumerate(hits):
                        if len(hit) > 0 and (sample_labels is None or sample_labels[j] not in LABEL_IGNORE):
                            overlap_sum[j] += weight

                if args.visualize:
                    nearby_scans.append((rname, i, pts))

            if args.fpfh_weight > 0 and len(sample_pts) >= 10:
                desc = _compute_fpfh_descriptiveness(sample_pts, args.voxel_leaf)
                overlap_sum *= 1.0 + args.fpfh_weight * desc

            # Normalize and save heatmap
            if overlap_sum.max() > 0:
                if args.normalize == "log":
                    vals = np.log1p(overlap_sum)
                elif args.normalize == "sqrt":
                    vals = np.sqrt(overlap_sum)
                else:
                    vals = overlap_sum
                normalized_overlap = vals / vals.max()
            else:
                normalized_overlap = np.zeros(len(sample_pts), dtype=np.float64)

            if overlap_sum.max() > 0:
                sample_scan_name = scan_files[sample_kf_idx]
                heatmap_path = os.path.join(heatmap_dir, sample_scan_name)
                tree = KDTree(sample_pts)
                _, nn_idx = tree.query(raw_sample_pts, k=1)
                nn_idx = np.atleast_1d(nn_idx.ravel())
                scan_heatmap = normalized_overlap[nn_idx].astype(np.float32)
                scan_heatmap.tofile(heatmap_path)

            if args.visualize:
                last_viz_data = {
                    "sample_pts": sample_pts,
                    "normalized_overlap": normalized_overlap,
                    "nearby_scans": nearby_scans,
                    "sample_kf_idx": sample_kf_idx,
                    "kf_ix": kf_ix,
                }

    if args.visualize and last_viz_data is not None:
        sample_pts = last_viz_data["sample_pts"]
        normalized_overlap = last_viz_data["normalized_overlap"]
        nearby_scans = last_viz_data["nearby_scans"]
        sample_kf_idx = last_viz_data["sample_kf_idx"]
        kf_ix = last_viz_data["kf_ix"]

        plotter = pv.Plotter()
        if args.heatmap and normalized_overlap is not None:
            cloud = pv.PolyData(sample_pts)
            cloud["overlap"] = normalized_overlap
            plotter.add_mesh(
                cloud,
                scalars="overlap",
                cmap="hot",
                scalar_bar_args={"title": "Overlap (normalized)"},
                point_size=2.0,
                render_points_as_spheres=False,
            )
            title = (
                f"Overlap heatmap: {args.sample_robot} kf={kf_ix} | "
                f"{len(nearby_scans)} nearby scans | voxel={args.voxel_leaf}m"
            )
        else:
            all_pts_list = [sample_pts]
            all_colors_list = [np.full((len(sample_pts), 3), [0, 1, 0])]
            for rname, idx, pts in nearby_scans:
                all_pts_list.append(pts)
                all_colors_list.append(np.full((len(pts), 3), [1, 0, 0]))
            all_pts = np.vstack(all_pts_list)
            all_colors = np.vstack(all_colors_list)
            cloud = pv.PolyData(all_pts)
            cloud["RGB"] = (all_colors * 255).astype(np.uint8)
            plotter.add_mesh(
                cloud,
                scalars="RGB",
                rgb=True,
                point_size=2.0,
                render_points_as_spheres=False,
            )
            title = (
                f"{args.sample_robot} kf={kf_ix} = green | "
                f"{len(nearby_scans)} nearby = red | voxel={args.voxel_leaf}m"
            )
        plotter.add_text(title, font_size=10, color="black")
        plotter.set_background("white")
        print("\nLaunching PyVista viewer...")
        plotter.show()


if __name__ == "__main__":
    main()
