#!/usr/bin/env python3
"""
Build and visualize a point cloud map from heatmap scans, colored by overlap value.

Loads scans that have corresponding heatmaps in heatmaps/, transforms each to world
frame, and displays in PyVista with a heat colormap (low overlap = cool, high = hot).

Usage:
    python3 plot_heatmap_map.py --env main_campus --robots robot1
    python3 plot_heatmap_map.py --env main_campus --robots robot1,robot2 --map-voxel-size 0.2
    python3 plot_heatmap_map.py --keyframe-dist 10  # Only keyframe scans

Requires: numpy, scipy, pyvista
"""

import argparse
import os
import glob

import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation

from utils import (
    _find_poses_csv, load_poses, apply_imu_to_lidar, get_keyframe_indices,
)


def _parse_args():
    p = argparse.ArgumentParser(
        description="Visualize heatmap scans as a colored map"
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
        "--robots",
        default="robot1",
        help="Comma-separated robot names",
    )
    p.add_argument(
        "--map-voxel-size",
        type=float,
        default=0.0,
        help="Voxel size (m) for final map downsampling (0=no voxel)",
    )
    p.add_argument(
        "--point-size",
        type=float,
        default=2.0,
        help="Point size for visualization",
    )
    p.add_argument(
        "--cmap",
        default="viridis",
        help="PyVista colormap name (e.g. hot, plasma, viridis)",
    )
    p.add_argument(
        "--keyframe-dist",
        type=float,
        default=0.0,
        help="If set, only load keyframe scans (min pose dist m). None = load all heatmap scans",
    )
    p.add_argument(
        "--min-intensity",
        type=float,
        default=None,
        help="Only show points with heatmap >= this value (0-1). Omit to show all.",
    )
    p.add_argument(
        "--max-intensity",
        type=float,
        default=None,
        help="Only show points with heatmap <= this value (0-1). Omit to show all.",
    )
    return p.parse_args()


def main():
    args = _parse_args()
    robot_names = [r.strip() for r in args.robots.split(",") if r.strip()]

    all_pts = []
    all_heatmap = []

    for rname in robot_names:
        rdir = os.path.join(args.data_dir, args.env, rname)
        heatmap_dir = os.path.join(rdir, "heatmaps")
        scan_bin_dir = os.path.join(rdir, "lidar_bin", "data")

        if not os.path.isdir(heatmap_dir):
            print(f"  WARNING: {heatmap_dir} not found, skipping {rname}")
            continue
        if not os.path.isdir(scan_bin_dir):
            print(f"  WARNING: {scan_bin_dir} not found, skipping {rname}")
            continue

        poses_csv = _find_poses_csv(rdir, rname, args.env)
        if not poses_csv or not os.path.isfile(poses_csv):
            print(f"  WARNING: no poses CSV in {rdir}, skipping {rname}")
            continue

        poses = load_poses(poses_csv)
        apply_imu_to_lidar(poses)

        scan_files = sorted(f for f in os.listdir(scan_bin_dir) if f.endswith(".bin"))
        n_avail = min(len(scan_files), len(poses))

        if args.keyframe_dist is not None:
            kf_indices = set(get_keyframe_indices(poses, n_avail, args.keyframe_dist))
        else:
            kf_indices = None

        heatmap_files = sorted(glob.glob(os.path.join(heatmap_dir, "*.bin")))

        n_loaded = 0
        for heatmap_path in heatmap_files:
            basename = os.path.basename(heatmap_path)
            try:
                idx = scan_files.index(basename)
            except ValueError:
                continue
            if idx >= len(poses):
                continue
            if kf_indices is not None and idx not in kf_indices:
                continue

            scan_path = os.path.join(scan_bin_dir, scan_files[idx])
            if not os.path.isfile(scan_path):
                continue

            pose = poses[idx]
            rot = Rotation.from_quat(pose[4:8])

            raw = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)
            pts = rot.apply(raw[:, :3].astype(np.float64)) + pose[1:4]

            heatmap = np.fromfile(heatmap_path, dtype=np.float32)
            if len(heatmap) != len(pts):
                print(f"    WARNING: {rname} idx {idx}: scan {len(pts)} pts vs heatmap {len(heatmap)}")
                continue

            all_pts.append(pts)
            all_heatmap.append(heatmap)
            n_loaded += 1

        print(f"  {rname}: {n_loaded} heatmap scans")

    if not all_pts:
        print("ERROR: no heatmap scans found. Run testing.py first to generate heatmaps.")
        return

    xyz = np.vstack(all_pts)
    heatmap_vals = np.concatenate(all_heatmap)

    # Filter by intensity range
    if args.min_intensity is not None or args.max_intensity is not None:
        mask = np.ones(len(heatmap_vals), dtype=bool)
        if args.min_intensity is not None:
            mask &= heatmap_vals >= args.min_intensity
        if args.max_intensity is not None:
            mask &= heatmap_vals <= args.max_intensity
        xyz = xyz[mask]
        heatmap_vals = heatmap_vals[mask]
        print(f"  Filtered by intensity: {len(xyz):,} points")
        # Renormalize to [0, 1] for full colormap range
        lo, hi = heatmap_vals.min(), heatmap_vals.max()
        if hi > lo:
            heatmap_vals = (heatmap_vals - lo) / (hi - lo)

    if args.map_voxel_size > 0:
        voxel = args.map_voxel_size
        keys = np.floor(xyz / voxel).astype(np.int64)
        _, ui = np.unique(keys, axis=0, return_index=True)
        xyz = xyz[ui]
        heatmap_vals = heatmap_vals[ui]
        print(f"  Voxelized: {len(xyz):,} points (voxel={voxel}m)")

    print(f"\nHeatmap range: [{heatmap_vals.min():.3f}, {heatmap_vals.max():.3f}]")

    cloud = pv.PolyData(xyz)
    cloud["heatmap"] = heatmap_vals

    plotter = pv.Plotter()
    plotter.add_mesh(
        cloud,
        scalars="heatmap",
        cmap=args.cmap,
        scalar_bar_args={"title": "Overlap"},
        point_size=args.point_size,
        render_points_as_spheres=False,
    )
    plotter.set_background("white")
    plotter.add_text(f"Heatmap map ({len(xyz):,} pts)", font_size=12, color="black")
    print("\nLaunching viewer...")
    plotter.show()


if __name__ == "__main__":
    main()
