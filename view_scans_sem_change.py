#!/usr/bin/env python3
"""
3D viewer combining DSM (LAS), DEM (TIF), OSM features, and multi-robot
lidar scans from CU_MULTI datasets.  Optionally projects OSM semantic labels
onto the accumulated scans and saves per-scan ground-truth labels.

Usage:
    # View DSM + OSM overlay with one robot's scans
    python3 view_DSM_DEM_OSM_SCANS.py --crs EPSG:2232 --crop-to-osm \\
        --data-dir /media/.../CU_MULTI --env kittredge_loop --robots robot1

    # Two robots, label + save
    python3 view_DSM_DEM_OSM_SCANS.py --crs EPSG:2232 --crop-to-osm \\
        --data-dir /media/.../CU_MULTI --env kittredge_loop \\
        --robots robot1,robot2 --icp --label --save-labels

Requires: laspy[lazrs], pyvista, pyproj, shapely, numpy, rasterio,
          scipy, osmium
"""

import argparse
import csv
import os
import re

import numpy as np
import laspy
import osmium
import pyvista as pv
from pyproj import CRS, Transformer
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
from shapely.geometry import Point, Polygon, LineString
from shapely import prepare
from shapely.strtree import STRtree

# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

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

# IMU → LiDAR extrinsic (CU_MULTI platform)
IMU_TO_LIDAR_T = np.array([-0.058038, 0.015573, 0.049603])
IMU_TO_LIDAR_Q = [0.0, 0.0, 1.0, 0.0]  # [qx, qy, qz, qw]

FT_TO_M = 0.3048006096

# LAS ASPRS classification colors
CLASS_COLORS = {
    0:  [180, 180, 180],   2: [139, 119,  80],   3: [144, 238, 144],
    4:  [ 34, 139,  34],   5: [  0, 100,   0],   6: [ 30, 120, 255],
    7:  [255,   0, 255],   9: [ 65, 105, 225],  11: [ 80,  80,  80],
    17: [192, 192, 192],
}

# Semantic label scheme
LABEL_NAMES = {
    0: "unlabeled",   1: "road",      2: "sidewalk",   3: "parking",
    4: "other-ground", 5: "building",  6: "fence",      7: "terrain",
    8: "water",        9: "bridge",   10: "vehicle",   11: "tree",
    12: "stairs",
}
LABEL_COLORS = {
    0:  [128, 128, 128],  1: [ 80,  80,  80],  2: [200, 160, 100],
    3:  [140, 100, 180],  4: [139,  90,  43],  5: [ 30, 120, 255],
    6:  [255, 165,   0],  7: [144, 238, 144],  8: [ 65, 105, 225],
    9:  [192, 192, 192], 10: [255,   0, 255], 11: [  0, 100,   0],
    12: [255, 128,   0],
}

# OSM tag classification helpers
SIDEWALK_HIGHWAY_TYPES = {"footway", "path", "pedestrian", "foot"}
ROAD_HIGHWAY_TYPES = {
    "primary", "secondary", "tertiary", "residential",
    "service", "unclassified", "trunk", "motorway",
    "primary_link", "secondary_link", "tertiary_link",
}
GRASSLAND_LANDUSE = {"grass", "meadow", "greenfield", "recreation_ground"}
GRASSLAND_NATURAL = {"grassland", "heath", "scrub"}
FOREST_NATURAL = {"wood", "forest"}
TREE_LANDUSE = {"orchard", "vineyard"}

OSM_COLORS = {
    "buildings":  [  0,  80, 200, 180], "roads":      [ 60,  60,  60, 200],
    "sidewalks":  [180, 140,  80, 180], "stairs":     [255, 128,   0, 200],
    "parking":    [120,  80, 160, 180], "grasslands": [100, 200, 100, 150],
    "forests":    [  0, 120,  30, 150], "water":      [ 40,  90, 200, 180],
}

ROAD_HALF_WIDTH_M = 4.0
SIDEWALK_HALF_WIDTH_M = 1.2
STAIRS_HALF_WIDTH_M = 1.5
FENCE_HALF_WIDTH_M = 0.5
TREE_POINT_RADIUS_M = 4.0
BUILDING_BUFFER_M = 2.0


# ═══════════════════════════════════════════════════════════════════════
# CRS helpers
# ═══════════════════════════════════════════════════════════════════════

def detect_feet(crs_obj):
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
    """Determine the native CRS of *las_files* (first file used)."""
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


# ═══════════════════════════════════════════════════════════════════════
# OSM parsing
# ═══════════════════════════════════════════════════════════════════════

def _parse_width(tags, fallback):
    raw = tags.get("width", "")
    if not raw:
        return fallback
    raw = raw.strip().lower().replace("m", "").replace(",", ".").strip()
    try:
        w = float(raw)
        if 0.1 < w < 200:
            return w
    except ValueError:
        pass
    return fallback


def _load_width_cache(path):
    cache = {}
    if path and os.path.isfile(path):
        try:
            with open(path) as f:
                for row in csv.DictReader(f):
                    cache[row["highway_type"]] = float(row["avg_width_m"])
        except Exception:
            pass
    return cache


def _save_width_cache(path, records):
    if not path or not records:
        return
    from collections import defaultdict
    by_type = defaultdict(list)
    for hw, w in records:
        by_type[hw].append(w)
    try:
        with open(path, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["highway_type", "avg_width_m", "count"])
            for hw in sorted(by_type):
                vals = by_type[hw]
                wr.writerow([hw, f"{sum(vals)/len(vals):.2f}", len(vals)])
    except Exception:
        pass


def _classify_tags(tags):
    """Map an OSM tag dict to a feature category string (or None)."""
    if "building" in tags:
        return "buildings"
    if tags.get("amenity") in ("parking", "parking_space"):
        return "parking"
    if tags.get("barrier") == "fence":
        return "fences"
    if tags.get("barrier") == "wall":
        return "walls"
    hw = tags.get("highway", "")
    if hw == "steps":
        return "stairs"
    if hw in SIDEWALK_HIGHWAY_TYPES:
        return "sidewalks"
    if hw in ROAD_HIGHWAY_TYPES:
        return "roads"
    if tags.get("natural") == "water":
        return "water"
    lu = tags.get("landuse", "")
    nat = tags.get("natural", "")
    if lu in GRASSLAND_LANDUSE or nat in GRASSLAND_NATURAL:
        return "grasslands"
    if tags.get("leisure") in ("park", "garden"):
        return "grasslands"
    if nat in FOREST_NATURAL or lu == "forest":
        return "forests"
    if lu in TREE_LANDUSE or tags.get("landcover") == "trees":
        return "trees_poly"
    return None


_LINE_CATS = {"roads", "sidewalks", "stairs", "fences", "walls"}
_POLY_CATS = {"buildings", "parking", "grasslands", "trees_poly",
              "forests", "water"}


class _OSMHandler(osmium.SimpleHandler):
    """Collects all OSM features needed for labelling and visualisation."""

    def __init__(self, width_cache=None):
        super().__init__()
        self.features = {
            "buildings": [], "roads": [], "sidewalks": [], "stairs": [],
            "parking": [], "grasslands": [], "trees_poly": [], "forests": [],
            "fences": [], "walls": [], "water": [], "tree_points": [],
        }
        self._wc = width_cache or {}
        self._last_road_w = ROAD_HALF_WIDTH_M * 2
        self._last_sw_w = SIDEWALK_HALF_WIDTH_M * 2
        self.width_records = []

    def node(self, n):
        if dict(n.tags).get("natural") == "tree" and n.location.valid():
            self.features["tree_points"].append((n.location.lat, n.location.lon))

    def way(self, w):
        tags = dict(w.tags)
        cat = _classify_tags(tags)
        if cat not in _LINE_CATS:
            return
        coords = [(nd.lat, nd.lon) for nd in w.nodes if nd.location.valid()]
        if not coords:
            return
        if cat in ("roads", "sidewalks"):
            self._road_sw(cat, tags, coords)
        elif cat == "stairs":
            self._stairs(tags, coords)
        else:
            self.features[cat].append(coords)

    def area(self, a):
        tags = dict(a.tags)
        cat = _classify_tags(tags)
        if cat not in _POLY_CATS:
            return
        for ring in a.outer_rings():
            coords = [(nd.lat, nd.lon) for nd in ring]
            if len(coords) >= 3:
                self.features[cat].append(coords)

    def _road_sw(self, cat, tags, coords):
        is_sw = cat == "sidewalks"
        last = self._last_sw_w if is_sw else self._last_road_w
        hw = tags.get("highway", "")
        if "width" in tags:
            w = _parse_width(tags, last)
            if is_sw:
                self._last_sw_w = w
            else:
                self._last_road_w = w
            self.width_records.append((hw, w))
        elif hw in self._wc:
            w = self._wc[hw]
        else:
            w = last
        self.features[cat].append((coords, w / 2.0))

    def _stairs(self, tags, coords):
        if "width" in tags:
            w = _parse_width(tags, STAIRS_HALF_WIDTH_M * 2)
        elif "steps" in self._wc:
            w = self._wc["steps"]
        else:
            w = STAIRS_HALF_WIDTH_M * 2
        self.width_records.append(("steps", w))
        self.features["stairs"].append((coords, w / 2.0))


def _parse_osm_bounds(path):
    """Extract <bounds> element from the first few lines of an .osm file."""
    try:
        with open(path) as f:
            for line in f:
                if "<bounds" in line:
                    m = re.search(
                        r'minlat="([^"]+)"\s+minlon="([^"]+)"\s+'
                        r'maxlat="([^"]+)"\s+maxlon="([^"]+)"', line)
                    if m:
                        return tuple(float(m.group(i)) for i in range(1, 5))
                if "<node" in line:
                    break
    except Exception:
        pass
    return None


def parse_osm_features(osm_path, width_cache_path=None):
    """Parse an .osm file and return (features_dict, bounds_or_None)."""
    wc = _load_width_cache(width_cache_path)
    h = _OSMHandler(wc)
    h.apply_file(osm_path, locations=True)
    bounds = _parse_osm_bounds(osm_path)
    counts = {k: len(v) for k, v in h.features.items() if v}
    print(f"    features: {counts}")
    if h.width_records:
        _save_width_cache(width_cache_path, h.width_records)
    return h.features, bounds


# ═══════════════════════════════════════════════════════════════════════
# OSM mesh building (for PyVista visualisation)
# ═══════════════════════════════════════════════════════════════════════

def _triangulate_polygon(poly):
    """Triangulate a Shapely polygon into a flat PyVista mesh (Z=0)."""
    from shapely import ops as sops
    try:
        if len(np.array(poly.exterior.coords)) < 4:
            return None
        tris = sops.triangulate(poly)
        pts, faces = [], []
        for tri in tris:
            if not poly.contains(tri.representative_point()):
                continue
            tc = np.array(tri.exterior.coords[:3])
            base = len(pts)
            pts.extend(tc.tolist())
            faces.extend([3, base, base + 1, base + 2])
        if not pts:
            return None
        pts3 = np.column_stack([np.array(pts)[:, :2],
                                np.zeros(len(pts))])
        return pv.PolyData(pts3, faces=np.array(faces, dtype=np.int32))
    except Exception:
        return None


def build_osm_meshes(features, to_crs_func):
    """Create flat 2D PyVista meshes from OSM features (caller sets Z)."""
    meshes = []

    def to_xy(lat, lon):
        return to_crs_func(lon, lat)

    for cat in ("grasslands", "forests", "water", "parking", "buildings"):
        rgba = OSM_COLORS.get(cat, [128, 128, 128, 160])
        for coords in features.get(cat, []):
            xy = [to_xy(lat, lon) for lat, lon in coords]
            if len(xy) < 3:
                continue
            try:
                poly = Polygon(xy)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if poly.is_empty:
                    continue
                m = _triangulate_polygon(poly)
                if m is not None:
                    meshes.append((m, rgba, cat))
            except Exception:
                pass

    for cat in ("roads", "sidewalks", "stairs"):
        rgba = OSM_COLORS.get(cat, [128, 128, 128, 160])
        for entry in features.get(cat, []):
            coords, half_w = entry
            xy = [to_xy(lat, lon) for lat, lon in coords]
            if len(xy) < 2:
                continue
            try:
                ls = LineString(xy)
                if not ls.is_valid or ls.is_empty:
                    continue
                buf = ls.buffer(half_w, cap_style="flat")
                if buf.is_empty:
                    continue
                m = _triangulate_polygon(buf)
                if m is not None:
                    meshes.append((m, rgba, cat))
            except Exception:
                pass

    print(f"  Built {len(meshes)} OSM meshes")
    return meshes


# ═══════════════════════════════════════════════════════════════════════
# DEM loader
# ═══════════════════════════════════════════════════════════════════════

def load_dem_surface(dem_path, native_crs, to_utm, is_feet, subsample=4):
    """Load a DEM .tif → UTM structured grid for PyVista."""
    import rasterio

    print(f"  Loading DEM from {dem_path}")
    with rasterio.open(dem_path) as src:
        data = src.read(1)
        nodata, tfm = src.nodata, src.transform
        nrows, ncols = data.shape

    ri = np.arange(0, nrows, subsample)
    ci = np.arange(0, ncols, subsample)
    rr, cc = np.meshgrid(ri, ci, indexing="ij")

    dx = tfm.c + cc * tfm.a + rr * tfm.b
    dy = tfm.f + cc * tfm.d + rr * tfm.e
    dz = data[ri][:, ::subsample].astype(np.float64)

    valid = np.isfinite(dz)
    if nodata is not None:
        valid &= dz != nodata
    if is_feet:
        dz *= FT_TO_M

    ux, uy = to_utm.transform(dx.ravel(), dy.ravel())
    ux = ux.reshape(dx.shape)
    uy = uy.reshape(dy.shape)
    valid &= np.isfinite(ux) & np.isfinite(uy)

    pts = np.column_stack([ux.ravel(), uy.ravel(), dz.ravel()])
    bad = ~valid.ravel()
    if bad.any():
        pts[bad, 2] = np.nanmedian(dz[valid]) if valid.any() else 0.0

    grid = pv.StructuredGrid()
    grid.dimensions = [dz.shape[1], dz.shape[0], 1]
    grid.points = pts
    print(f"    grid {dz.shape[0]}×{dz.shape[1]}, "
          f"Z(m): [{dz[valid].min():.2f}, {dz[valid].max():.2f}]")
    return grid


# ═══════════════════════════════════════════════════════════════════════
# Spatial index + semantic labelling
# ═══════════════════════════════════════════════════════════════════════

def build_spatial_index(features, to_crs_func):
    """Build Shapely STRtree indices from OSM features (lat/lon → UTM)."""
    idx = {}

    def xy(lat, lon):
        return to_crs_func(lon, lat)

    for cat in ("buildings", "parking", "grasslands", "trees_poly",
                "forests", "water"):
        geoms = []
        for coords in features.get(cat, []):
            pts = [xy(la, lo) for la, lo in coords]
            if len(pts) < 3:
                continue
            try:
                p = Polygon(pts)
                if not p.is_valid:
                    p = p.buffer(0)
                if p.is_empty:
                    continue
                if cat == "buildings":
                    p = p.buffer(BUILDING_BUFFER_M)
                prepare(p)
                geoms.append(p)
            except Exception:
                pass
        idx[cat] = (STRtree(geoms), geoms) if geoms else (None, [])

    for cat in ("roads", "sidewalks", "stairs"):
        geoms = []
        for coords, hw in features.get(cat, []):
            pts = [xy(la, lo) for la, lo in coords]
            if len(pts) < 2:
                continue
            try:
                ls = LineString(pts)
                if ls.is_valid and not ls.is_empty:
                    buf = ls.buffer(hw, cap_style="flat")
                    if not buf.is_empty:
                        prepare(buf)
                        geoms.append(buf)
            except Exception:
                pass
        idx[cat] = (STRtree(geoms), geoms) if geoms else (None, [])

    for cat in ("fences", "walls"):
        geoms = []
        for coords in features.get(cat, []):
            pts = [xy(la, lo) for la, lo in coords]
            if len(pts) < 2:
                continue
            try:
                ls = LineString(pts)
                if ls.is_valid and not ls.is_empty:
                    geoms.append(ls)
            except Exception:
                pass
        idx[cat] = (STRtree(geoms), geoms) if geoms else (None, [])

    tp = [Point(xy(la, lo)) for la, lo in features.get("tree_points", [])]
    idx["tree_points"] = (STRtree(tp), tp) if tp else (None, [])
    return idx


def label_points(x, y, z, classification, spatial_idx,
                 ground_z_func=None, tree_height_thresh=1.5,
                 vehicle_max_height=2.0, fence_max_height=1.5):
    """Assign semantic labels (UTM metres). Three phases:
    1) Hard LAS classification   2) OSM spatial queries   3) Cross-validate
    """
    n = len(x)
    labels = np.zeros(n, dtype=np.int32)

    if ground_z_func is not None:
        gz = ground_z_func(x, y)
        hag = z - gz
        has_g = np.isfinite(hag)
        print(f"  Height-above-ground: {has_g.sum():,}/{n:,} have ground ref")
    else:
        m2 = classification == 2
        fb = np.median(z[m2]) if m2.any() else np.percentile(z, 5)
        hag = z - fb
        has_g = np.ones(n, dtype=bool)
        print(f"  No ground model — using fallback Z={fb:.2f}")

    at_gnd = has_g & (hag < tree_height_thresh)
    above = has_g & (hag >= tree_height_thresh)

    # Phase 1 — LAS classification
    labels[classification == 17] = 9
    labels[classification == 20] = 4
    labels[classification == 9] = 8
    labels[classification == 11] = 1
    labels[classification == 6] = 5
    veg = (classification == 3) | (classification == 4) | (classification == 5)
    veg_tree = veg & above
    labels[veg_tree] = 11
    done = labels != 0
    print(f"  Phase 1 (LAS): {done.sum():,} labelled")

    # Phase 2 — OSM spatial queries
    def _contains(cat, px, py):
        t, gs = spatial_idx.get(cat, (None, []))
        if t is None:
            return False
        pt = Point(px, py)
        return any(gs[i].contains(pt) for i in t.query(pt))

    def _near(cat, px, py, buf):
        t, gs = spatial_idx.get(cat, (None, []))
        if t is None:
            return False
        pt = Point(px, py)
        area = pt.buffer(buf)
        return any(gs[i].distance(pt) <= buf for i in t.query(area))

    todo = np.where(~done)[0]
    total = len(todo)
    bs = 50000
    for s in range(0, total, bs):
        if s > 0 and s % (bs * 5) == 0:
            print(f"    {100*s/total:.0f}%")
        for i in todo[s:s + bs]:
            px, py = x[i], y[i]
            g = bool(at_gnd[i])
            if _contains("buildings", px, py):
                labels[i] = 5; continue
            if _contains("water", px, py):
                labels[i] = 8; continue
            if _contains("stairs", px, py):
                labels[i] = 12; continue
            if g and _contains("roads", px, py):
                labels[i] = 1; continue
            if g and _contains("sidewalks", px, py):
                labels[i] = 2; continue
            if _contains("parking", px, py):
                h = hag[i] if has_g[i] else 0.0
                if g:
                    labels[i] = 3; continue
                elif h <= vehicle_max_height:
                    labels[i] = 10; continue
            if _near("fences", px, py, FENCE_HALF_WIDTH_M):
                h = hag[i] if has_g[i] else 0.0
                if h <= fence_max_height:
                    labels[i] = 6
                continue
            if _near("walls", px, py, FENCE_HALF_WIDTH_M):
                h = hag[i] if has_g[i] else 0.0
                if h <= fence_max_height:
                    labels[i] = 6
                continue
            if (_contains("grasslands", px, py) or
                    _contains("trees_poly", px, py) or
                    _contains("forests", px, py)):
                labels[i] = 7 if g else 11; continue
            if _near("tree_points", px, py, TREE_POINT_RADIUS_M):
                labels[i] = 11; continue
            if g:
                labels[i] = 4
            elif above[i]:
                labels[i] = 11
    print(f"    100%")

    # Phase 3 — tree inside building → building
    up = 0
    for i in np.where(veg_tree)[0]:
        if _contains("buildings", x[i], y[i]):
            labels[i] = 5; up += 1
    if up:
        print(f"  Phase 3: {up:,} tree→building")
    return labels


# ═══════════════════════════════════════════════════════════════════════
# Kernel-based semantic change detection
# ═══════════════════════════════════════════════════════════════════════

def detect_semantic_changes(x, y, labels, kernel_radius):
    """Detect points where two or more semantic classes meet.

    For each labeled point, queries all neighbors within kernel_radius.
    If 2+ distinct non-zero labels exist in that neighborhood, the point
    is marked as a semantic change point.

    Returns a boolean mask (True = semantic change).
    """
    n = len(x)
    xy = np.column_stack([x, y])

    labeled_mask = labels > 0
    labeled_idx = np.where(labeled_mask)[0]
    if len(labeled_idx) == 0:
        print("  No labeled points — skipping semantic change detection")
        return np.zeros(n, dtype=bool)

    print(f"  Building KDTree for {len(labeled_idx):,} labeled points...")
    tree = KDTree(xy[labeled_idx])

    print(f"  Querying neighbors within kernel_radius={kernel_radius:.2f} m...")
    sem_change = np.zeros(n, dtype=bool)

    batch_size = 50000
    total = len(labeled_idx)
    n_changes = 0

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_pts = xy[labeled_idx[start:end]]

        neighbors_list = tree.query_ball_point(batch_pts, kernel_radius)

        for local_i, neighbors in enumerate(neighbors_list):
            if len(neighbors) < 2:
                continue
            neighbor_labels = labels[labeled_idx[neighbors]]
            unique = np.unique(neighbor_labels[neighbor_labels > 0])
            if len(unique) >= 2:
                global_i = labeled_idx[start + local_i]
                sem_change[global_i] = True
                n_changes += 1

        if start > 0 and start % (batch_size * 5) == 0:
            print(f"    {100 * start / total:.0f}%")

    print(f"    100%")
    print(f"  Semantic change points: {n_changes:,} / {total:,} "
          f"({100 * n_changes / total:.1f}%)")

    return sem_change


# ═══════════════════════════════════════════════════════════════════════
# FPFH (Fast Point Feature Histograms)
# ═══════════════════════════════════════════════════════════════════════

def _estimate_normals(pts, k=30):
    """Estimate surface normals via PCA of k-nearest neighborhoods."""
    tree = KDTree(pts)
    _, idx = tree.query(pts, k=min(k, len(pts)))
    normals = np.zeros_like(pts)
    for i in range(len(pts)):
        neighbors = pts[idx[i]]
        centered = neighbors - neighbors.mean(axis=0)
        cov = centered.T @ centered
        eigvals, eigvecs = np.linalg.eigh(cov)
        normals[i] = eigvecs[:, 0]
    # Orient normals consistently (upward preference)
    flip = normals[:, 2] < 0
    normals[flip] *= -1
    return normals


def _spfh(pts, normals, tree, idx_lists, n_bins=11):
    """Compute SPFH (Simplified PFH) for each point.

    For each (point, neighbor) pair, computes three angular features
    (alpha, phi, theta) and bins them into histograms.
    Returns (N, 3*n_bins) array.
    """
    n = len(pts)
    histograms = np.zeros((n, 3 * n_bins), dtype=np.float64)

    for i in range(n):
        neighbors = idx_lists[i]
        if len(neighbors) < 2:
            continue

        n_i = normals[i]
        p_i = pts[i]
        alphas, phis, thetas = [], [], []

        for j in neighbors:
            if j == i:
                continue
            delta = pts[j] - p_i
            d = np.linalg.norm(delta)
            if d < 1e-12:
                continue

            u = n_i
            v = np.cross(delta / d, u)
            v_norm = np.linalg.norm(v)
            if v_norm < 1e-12:
                continue
            v = v / v_norm
            w = np.cross(u, v)

            n_j = normals[j]
            alphas.append(np.dot(v, n_j))
            phis.append(np.dot(u, delta / d))
            thetas.append(np.arctan2(np.dot(w, n_j), np.dot(u, n_j)))

        if not alphas:
            continue

        ha, _ = np.histogram(alphas, bins=n_bins, range=(-1, 1))
        hp, _ = np.histogram(phis, bins=n_bins, range=(-1, 1))
        ht, _ = np.histogram(thetas, bins=n_bins, range=(-np.pi, np.pi))

        total = len(alphas)
        histograms[i, :n_bins] = ha / total
        histograms[i, n_bins:2*n_bins] = hp / total
        histograms[i, 2*n_bins:] = ht / total

    return histograms


def compute_fpfh(pts, radius, n_bins=11, normal_k=30):
    """Compute FPFH descriptors for a 3D point cloud.

    1. Estimate normals via local PCA
    2. Build SPFH for each point using neighbors within radius
    3. Weight neighbor SPFHs to produce final FPFH

    Returns (N, 3*n_bins) array of FPFH descriptors.
    """
    n = len(pts)
    print(f"  Estimating normals (k={normal_k})...")
    normals = _estimate_normals(pts, k=normal_k)

    print(f"  Building KDTree and querying radius={radius:.2f} m...")
    tree = KDTree(pts)
    idx_lists = tree.query_ball_point(pts, radius)

    print(f"  Computing SPFH...")
    spfh = _spfh(pts, normals, tree, idx_lists, n_bins)

    print(f"  Computing FPFH (weighting neighbor SPFHs)...")
    fpfh = np.zeros_like(spfh)
    for i in range(n):
        neighbors = idx_lists[i]
        if len(neighbors) < 2:
            fpfh[i] = spfh[i]
            continue

        weighted_sum = np.zeros(3 * n_bins)
        weight_total = 0.0
        p_i = pts[i]

        for j in neighbors:
            if j == i:
                continue
            d = np.linalg.norm(pts[j] - p_i)
            if d < 1e-12:
                continue
            w = 1.0 / d
            weighted_sum += w * spfh[j]
            weight_total += w

        if weight_total > 0:
            fpfh[i] = spfh[i] + weighted_sum / weight_total
        else:
            fpfh[i] = spfh[i]

    print(f"  FPFH shape: {fpfh.shape}")
    return fpfh


def fpfh_to_rgb(fpfh):
    """Map FPFH descriptors to RGB colors via PCA (33D → 3D)."""
    centered = fpfh - fpfh.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    top3 = eigvecs[:, -3:][:, ::-1]
    reduced = centered @ top3

    rgb = np.zeros((len(fpfh), 3), dtype=np.uint8)
    for i in range(3):
        col = reduced[:, i]
        mn, mx = col.min(), col.max()
        if mx > mn:
            rgb[:, i] = ((col - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            rgb[:, i] = 128
    return rgb


# ═══════════════════════════════════════════════════════════════════════
# Z-only ICP
# ═══════════════════════════════════════════════════════════════════════

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
    # Final error
    shifted = source.copy()
    shifted[:, 2] += dz_total
    d2, idx = tree.query(shifted[:, :2])
    ok = d2 < max_correspondence_dist
    err = np.mean(np.abs(target[idx[ok], 2] - shifted[ok, 2])) if ok.sum() else float("inf")
    return dz_total, err, int(ok.sum())


# ═══════════════════════════════════════════════════════════════════════
# Scan loading helpers
# ═══════════════════════════════════════════════════════════════════════

def _find_poses_csv(robot_dir, robot_name, env_name):
    """Locate the poses CSV inside *robot_dir*."""
    # Convention: {robot}_{env}_gt_utm_poses.csv
    expected = os.path.join(robot_dir,
                            f"{robot_name}_{env_name}_gt_utm_poses.csv")
    if os.path.isfile(expected):
        return expected
    # Fallback: any *_utm_poses.csv
    for f in os.listdir(robot_dir):
        if f.endswith("_utm_poses.csv"):
            return os.path.join(robot_dir, f)
    # Last resort
    fb = os.path.join(robot_dir, "poses.csv")
    return fb if os.path.isfile(fb) else None


def load_poses(csv_path):
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


def apply_imu_to_lidar(poses):
    """Transform IMU-frame poses to LiDAR-frame in-place."""
    rot_ext = Rotation.from_quat(IMU_TO_LIDAR_Q).as_matrix()
    for i in range(len(poses)):
        R_imu = Rotation.from_quat(poses[i, 4:8]).as_matrix()
        poses[i, 1:4] += R_imu @ IMU_TO_LIDAR_T
        poses[i, 4:8] = Rotation.from_matrix(R_imu @ rot_ext).as_quat()


def accumulate_robot_scans(robot_dir, robot_name, env_name, args,
                           dsm_tree_2d, dsm_xyz):
    """Load, ICP-align, voxelise and accumulate scans for one robot.

    Returns (scan_pts, trajectory, z_shifts, indices, poses, scan_files,
             scan_bin_dir) or Nones on failure.
    """
    poses_csv = _find_poses_csv(robot_dir, robot_name, env_name)
    scan_bin_dir = os.path.join(robot_dir, "lidar_bin", "data")

    if not poses_csv or not os.path.isdir(scan_bin_dir):
        if not poses_csv:
            print(f"    WARNING: no poses CSV in {robot_dir}")
        if not os.path.isdir(scan_bin_dir):
            print(f"    WARNING: no lidar_bin/data/ in {robot_dir}")
        return None

    print(f"  Poses: {os.path.basename(poses_csv)}")
    poses = load_poses(poses_csv)
    print(f"  {len(poses)} poses loaded")
    apply_imu_to_lidar(poses)
    print(f"  Applied IMU → LiDAR transform")

    scan_files = sorted(f for f in os.listdir(scan_bin_dir)
                        if f.endswith(".bin"))
    n_avail = min(len(scan_files), len(poses))

    # Select keyframes by minimum distance between consecutive poses
    kf_dist = args.keyframe_dist
    indices = [0]
    last_pos = poses[0, 1:4].copy()
    for i in range(1, n_avail):
        pos = poses[i, 1:4]
        if np.linalg.norm(pos - last_pos) >= kf_dist:
            indices.append(i)
            last_pos = pos.copy()
            if args.scan_max is not None and len(indices) >= args.scan_max:
                break
    print(f"  {n_avail} scans available → {len(indices)} keyframes "
          f"(dist≥{kf_dist}m)")

    all_pts, traj, z_shifts = [], [], []
    accepted_indices = []
    skipped = 0
    accum_tree, accum_xyz = None, None
    voxel = args.scan_voxel_size
    radius = args.icp_radius
    z_thresh = args.icp_z_thresh
    z_window = args.icp_z_window
    z_retries = args.icp_z_retries

    for count, idx in enumerate(indices):
        pose = poses[idx]
        tx, ty, tz = pose[1], pose[2], pose[3]
        rot = Rotation.from_quat(pose[4:8])

        raw = np.fromfile(os.path.join(scan_bin_dir, scan_files[idx]),
                          dtype=np.float32).reshape(-1, 4)
        pts = rot.apply(raw[:, :3].astype(np.float64)) + [tx, ty, tz]

        # Per-scan Z-ICP with consistency check
        zs = 0.0
        scan_accepted = True
        if args.icp and dsm_tree_2d is not None and len(pts) > 100:
            ctr = pts[:, :2].mean(axis=0)
            parts = []
            di = dsm_tree_2d.query_ball_point(ctr, radius)
            if di:
                parts.append(dsm_xyz[di])
            if accum_tree is not None:
                ai = accum_tree.query_ball_point(ctr, radius)
                if ai:
                    parts.append(accum_xyz[ai])
            if parts:
                tgt = np.vstack(parts)
                if len(tgt) > 100:
                    sub = pts[::4] if len(pts) > 400 else pts
                    recent = z_shifts[-z_window:] if z_shifts else []

                    for attempt in range(max(1, z_retries)):
                        zs, _, _ = icp_z_only(
                            sub, tgt,
                            max_iterations=args.icp_max_iter,
                            max_correspondence_dist=args.icp_max_corr_dist)
                        if recent:
                            ref = np.median(recent)
                            if abs(zs - ref) <= z_thresh:
                                break
                        else:
                            break
                    else:
                        if recent and abs(zs - np.median(recent)) > z_thresh:
                            skipped += 1
                            scan_accepted = False

        if not scan_accepted:
            continue

        pts[:, 2] += zs
        z_shifts.append(zs)
        accepted_indices.append(idx)

        if voxel > 0:
            keys = np.floor(pts / voxel).astype(np.int64)
            _, ui = np.unique(keys, axis=0, return_index=True)
            pts = pts[ui]

        all_pts.append(pts)
        traj.append([tx, ty, tz + zs])

        loaded = count + 1
        if args.icp and loaded % 50 == 0:
            accum_xyz = np.vstack(all_pts)
            accum_tree = KDTree(accum_xyz[:, :2])
        if loaded % 50 == 0 or loaded == len(indices):
            n = sum(len(p) for p in all_pts)
            zs_arr = np.array(z_shifts) if z_shifts else np.array([0.0])
            print(f"    [{loaded}/{len(indices)}] {n:,} pts | "
                  f"z_shift: {zs_arr.mean():+.3f}±{zs_arr.std():.3f} m"
                  f"{f' | {skipped} skipped' if skipped else ''}")

    if not all_pts:
        print(f"    WARNING: all scans skipped for {robot_name}")
        return None

    if skipped:
        print(f"  {skipped} scans skipped (Z-shift exceeded threshold)")

    scan_pts = np.vstack(all_pts)
    trajectory = np.array(traj)
    return {
        "pts": scan_pts, "traj": trajectory, "z_shifts": z_shifts,
        "indices": accepted_indices, "poses": poses,
        "scan_files": scan_files,
        "scan_bin_dir": scan_bin_dir, "robot": robot_name,
    }


# ═══════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="DSM + DEM + OSM + multi-robot scan viewer / labeller")

    # Data sources
    p.add_argument("--las", default="N2W215.las,N2W225.las",
                   help="Comma-separated LAS files (relative to script dir)")
    p.add_argument("--dem", default="N2W215.tif,N2W225.tif",
                   help="Comma-separated DEM .tif files")
    p.add_argument("--osm",
                   default="north_campus.osm,main_campus.osm,kittredge_loop.osm",
                   help="Comma-separated OSM files")
    p.add_argument("--crs", default=None, help="Override CRS (e.g. EPSG:2232)")

    # Robot scan data
    p.add_argument("--data-dir",
                   default="/media/donceykong/doncey_ssd_02/datasets/CU_MULTI",
                   help="Root dataset directory")
    p.add_argument("--env", default="main_campus",
                   help="Environment name (sub-folder under data-dir)")
    p.add_argument("--robots", default="robot1",
                   help="Comma-separated robot names")

    # Scan selection / processing
    p.add_argument("--keyframe-dist", type=float, default=15.0,
                   help="Min distance (m) between keyframes (pose-based)")
    p.add_argument("--scan-max", type=int, default=None,
                   help="Max number of scans to load")
    p.add_argument("--scan-voxel-size", type=float, default=0.5,
                   help="Voxel size (m) for per-scan downsampling")

    # ICP
    p.add_argument("--icp", action="store_true", default=True,
                   help="Per-scan Z-only ICP (scan → DSM + accumulated)")
    p.add_argument("--icp-max-iter", type=int, default=100)
    p.add_argument("--icp-max-corr-dist", type=float, default=5.0)
    p.add_argument("--icp-radius", type=float, default=50.0,
                   help="Radius (m) around scan centre for ICP target")
    p.add_argument("--icp-z-thresh", type=float, default=1.0,
                   help="Max allowed Z-shift deviation (m) from recent scans")
    p.add_argument("--icp-z-window", type=int, default=10,
                   help="Number of recent Z-shifts to compare against")
    p.add_argument("--icp-z-retries", type=int, default=500,
                   help="Retry attempts when Z-shift exceeds threshold")

    # Labelling
    p.add_argument("--label", action="store_true", default=True,
                   help="Project OSM semantics onto scans (colour by class)")
    p.add_argument("--save-labels", action="store_true", default=False,
                   help="Save per-scan .label files to groundtruth_labels/")
    p.add_argument("--tree-height", type=float, default=0.25,
                   help="Height (m) threshold for tree vs terrain")
    p.add_argument("--vehicle-max-height", type=float, default=2.0,
                   help="Max height (m) above ground for vehicle label")
    p.add_argument("--fence-max-height", type=float, default=1.5,
                   help="Max height (m) above ground for fence label")
    p.add_argument("--width-cache", default=None,
                   help="Road/sidewalk width cache CSV")
    p.add_argument("--nn-max-dist", type=float, default=2.0,
                   help="Max NN distance (m) for label transfer to scans")
    p.add_argument("--nearest-dsm-neighbor", type=float, default=3.0,
                   help="Max DSM NN dist (m); farther points are unlabeled "
                        "unless road/sidewalk/terrain/tree/parking/stairs")
    p.add_argument("--kernel-radius", type=float, default=0.5,
                   help="Kernel radius (m) for semantic change detection")
    p.add_argument("--fpfh", action="store_true", default=False,
                   help="Compute and visualize FPFH descriptors")
    p.add_argument("--fpfh-radius", type=float, default=1.0,
                   help="Radius (m) for FPFH neighbor search")

    # Visualisation toggles
    p.add_argument("--las-subsample", type=int, default=8)
    p.add_argument("--dem-subsample", type=int, default=4)
    p.add_argument("--crop-to-osm", action="store_true")
    p.add_argument("--point-size", type=float, default=2.0)
    p.add_argument("--scan-point-size", type=float, default=2.0)
    p.add_argument("--osm-z-offset", type=float, default=-500.0)
    p.add_argument("--show-dsm", action="store_true", default=False)
    p.add_argument("--no-dsm", dest="show_dsm", action="store_false")
    p.add_argument("--dsm-color-by-height", action="store_true",
                   default=False, help="Color DSM point cloud by elevation")
    p.add_argument("--show-dem", action="store_true", default=False)
    p.add_argument("--no-dem", dest="show_dem", action="store_false")
    p.add_argument("--dem-color-by-height", action="store_true",
                   default=False, help="Color DEM surface by elevation")
    p.add_argument("--show-osm", action="store_true", default=False)
    p.add_argument("--no-osm", dest="show_osm", action="store_false")
    p.add_argument("--show-scans", action="store_true", default=False)
    p.add_argument("--no-scans", dest="show_scans", action="store_false")
    p.add_argument("--show-trajectory", action="store_true", default=False)
    p.add_argument("--no-trajectory", dest="show_trajectory",
                   action="store_false")
    p.add_argument("--show-sem-change-only", action="store_true", default=False,
                   help="Display only semantic change points")
    p.add_argument("--save", default=None, help="Save screenshot path")

    args = p.parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # ── Resolve file paths ────────────────────────────────────────────
    def _resolve(names_csv, base):
        return [p if os.path.isabs(p) else os.path.join(base, p)
                for p in (s.strip() for s in names_csv.split(",")) if p]

    las_paths = _resolve(args.las, script_dir)
    dem_paths = _resolve(args.dem, script_dir)
    osm_paths = _resolve(args.osm, script_dir)

    robot_names = [r.strip() for r in args.robots.split(",") if r.strip()]
    robot_dirs = [os.path.join(args.data_dir, args.env, r)
                  for r in robot_names]

    # ══════════════════════════════════════════════════════════════════
    # 1. Load LAS files
    # ══════════════════════════════════════════════════════════════════
    print("Loading LAS files...")
    las_files = []
    for lp in las_paths:
        if not os.path.isfile(lp):
            print(f"  WARNING: {lp} not found"); continue
        las = laspy.read(lp)
        print(f"  {os.path.basename(lp)}: {len(las.points):,} pts")
        las_files.append(las)
    if not las_files:
        print("ERROR: no LAS files."); return

    # ── CRS ──
    print("\nResolving CRS...")
    native_crs = resolve_crs(las_files, args.crs)
    if native_crs is None:
        print("  ERROR: could not resolve CRS"); return
    is_feet = detect_feet(native_crs)
    print(f"  feet={is_feet}")

    utm_crs = CRS(_UTM13N)
    native_to_utm = Transformer.from_crs(native_crs, utm_crs, always_xy=True)
    wgs_to_utm = Transformer.from_crs(_WGS84_P4, utm_crs, always_xy=True)

    # ══════════════════════════════════════════════════════════════════
    # 2. Parse OSM
    # ══════════════════════════════════════════════════════════════════
    osm_features, osm_bounds = None, None
    print("\nParsing OSM files...")
    wcache = args.width_cache or os.path.join(script_dir, "osm_widths.csv")
    for op in osm_paths:
        if not os.path.isfile(op):
            print(f"  WARNING: {op} not found"); continue
        print(f"  {os.path.basename(op)}")
        feats, bnds = parse_osm_features(op, wcache)
        if osm_features is None:
            osm_features = feats
        else:
            for k in osm_features:
                osm_features[k].extend(feats.get(k, []))
        if bnds:
            if osm_bounds is None:
                osm_bounds = bnds
            else:
                osm_bounds = (min(osm_bounds[0], bnds[0]),
                              min(osm_bounds[1], bnds[1]),
                              max(osm_bounds[2], bnds[2]),
                              max(osm_bounds[3], bnds[3]))
    if osm_features:
        tot = {k: len(v) for k, v in osm_features.items() if v}
        print(f"  Combined: {tot}")

    # ── Crop box ──
    crop_box = None
    if args.crop_to_osm and osm_bounds:
        to_nat = Transformer.from_crs(_WGS84_P4, native_crs, always_xy=True)
        pad = 0.001
        sw = to_nat.transform(osm_bounds[1] - pad, osm_bounds[0] - pad)
        ne = to_nat.transform(osm_bounds[3] + pad, osm_bounds[2] + pad)
        crop_box = (sw[0], sw[1], ne[0], ne[1])

    # ══════════════════════════════════════════════════════════════════
    # 3. Extract LAS → UTM
    # ══════════════════════════════════════════════════════════════════
    print("\nExtracting LAS → UTM...")
    ax, ay, az, acls = [], [], [], []
    for i, las in enumerate(las_files):
        sub = args.las_subsample
        lx = np.asarray(las.x[::sub], dtype=np.float64)
        ly = np.asarray(las.y[::sub], dtype=np.float64)
        lz = np.asarray(las.z[::sub], dtype=np.float64)
        lc = np.asarray(las.classification[::sub])
        if crop_box:
            m = ((lx >= crop_box[0]) & (lx <= crop_box[2]) &
                 (ly >= crop_box[1]) & (ly <= crop_box[3]))
            if m.sum() > 0:
                lx, ly, lz, lc = lx[m], ly[m], lz[m], lc[m]
                print(f"  {os.path.basename(las_paths[i])}: "
                      f"{len(lx):,} pts (cropped)")
            else:
                print(f"  {os.path.basename(las_paths[i])}: "
                      f"{len(lx):,} pts (kept)")
        else:
            print(f"  {os.path.basename(las_paths[i])}: {len(lx):,} pts")
        ux, uy = native_to_utm.transform(lx, ly)
        uz = lz * FT_TO_M if is_feet else lz.copy()
        ax.append(ux); ay.append(uy); az.append(uz); acls.append(lc)

    x = np.concatenate(ax); y = np.concatenate(ay)
    z = np.concatenate(az); cls = np.concatenate(acls)
    print(f"  Total: {len(x):,} pts  "
          f"X:[{x.min():.0f},{x.max():.0f}] "
          f"Y:[{y.min():.0f},{y.max():.0f}] "
          f"Z:[{z.min():.1f},{z.max():.1f}]")

    # ── Ground surface ──
    print("\nBuilding ground surface...")
    gm = cls == 2
    ground_z_func = None
    if gm.sum() > 100:
        gx, gy, gz_g = x[gm][::4], y[gm][::4], z[gm][::4]
        g_tree = KDTree(np.column_stack([gx, gy]))
        def ground_z_func(qx, qy):
            d, ix = g_tree.query(np.column_stack([qx, qy]))
            r = gz_g[ix].copy(); r[d > 50] = np.nan; return r
        print(f"  {len(gx):,} ground pts")
    else:
        print("  WARNING: too few ground points")

    # ══════════════════════════════════════════════════════════════════
    # 4. Load DEM surfaces
    # ══════════════════════════════════════════════════════════════════
    dem_grids = []
    if args.show_dem:
        print("\nLoading DEM → UTM...")
        for dp in dem_paths:
            if not os.path.isfile(dp):
                continue
            try:
                dem_grids.append(
                    load_dem_surface(dp, native_crs, native_to_utm,
                                    is_feet, args.dem_subsample))
            except Exception as e:
                print(f"  WARNING: {dp}: {e}")

    # ══════════════════════════════════════════════════════════════════
    # 5. Build OSM visualisation meshes
    # ══════════════════════════════════════════════════════════════════
    osm_meshes = []
    if args.show_osm and osm_features:
        print("\nBuilding OSM meshes...")
        osm_meshes = build_osm_meshes(osm_features, wgs_to_utm.transform)

    # ══════════════════════════════════════════════════════════════════
    # 6. Load and accumulate scans from all robots
    # ══════════════════════════════════════════════════════════════════
    robot_data = []          # per-robot metadata for saving
    all_robot_pts = []       # combined for visualisation / labelling
    all_robot_traj = []

    if args.show_scans or args.save_labels:
        # Pre-build DSM KDTree for ICP
        dsm_tree_2d, dsm_xyz = None, None
        if args.icp and len(x) > 0:
            dsm_xyz = np.column_stack([x, y, z])
            dsm_tree_2d = KDTree(dsm_xyz[:, :2])
            print(f"\nDSM KDTree: {len(dsm_xyz):,} pts")

        for rname, rdir in zip(robot_names, robot_dirs):
            print(f"\n{'─'*60}")
            print(f"Robot: {rname}  ({rdir})")
            print(f"{'─'*60}")
            if not os.path.isdir(rdir):
                print(f"  WARNING: directory not found"); continue
            result = accumulate_robot_scans(
                rdir, rname, args.env, args, dsm_tree_2d, dsm_xyz)
            if result is None:
                continue
            robot_data.append(result)
            all_robot_pts.append(result["pts"])
            all_robot_traj.append(result["traj"])
            print(f"  {rname}: {result['pts'].shape[0]:,} pts, "
                  f"{len(result['indices'])} scans")

    scan_pts_utm = (np.vstack(all_robot_pts) if all_robot_pts
                    else None)
    trajectory_utm = (np.vstack(all_robot_traj) if all_robot_traj
                      else None)

    # ══════════════════════════════════════════════════════════════════
    # 7. Semantic labelling
    # ══════════════════════════════════════════════════════════════════
    scan_labels = None
    sem_change_mask = None
    fpfh_colors = None
    if (args.label or args.save_labels) and scan_pts_utm is not None:
        print("\n" + "=" * 60)
        print("Semantic labelling")
        print("=" * 60)

        # NN-transfer DSM classifications
        print("\nTransferring DSM classifications via NN...")
        dsm_nn = KDTree(np.column_stack([x, y]))
        dsm_d, dsm_ix = dsm_nn.query(scan_pts_utm[:, :2])
        nn_cls = cls[dsm_ix].copy()
        nn_cls[dsm_d > 10.0] = 0
        nn_cls[(nn_cls == 7) | (nn_cls == 18)] = 0
        print(f"  {(dsm_d <= 10).sum():,} within 10 m")

        # Spatial index
        if osm_features:
            print("\nBuilding OSM spatial index (UTM)...")
            spatial_idx = build_spatial_index(osm_features,
                                             wgs_to_utm.transform)
        else:
            spatial_idx = {}

        # Label
        print("\nLabelling...")
        sx = scan_pts_utm[:, 0].astype(np.float64)
        sy = scan_pts_utm[:, 1].astype(np.float64)
        sz = scan_pts_utm[:, 2].astype(np.float64)
        scan_labels = label_points(sx, sy, sz, nn_cls, spatial_idx,
                                   ground_z_func, args.tree_height,
                                   args.vehicle_max_height,
                                   args.fence_max_height)

        # DSM proximity filter: points far from DSM keep label only if
        # road/sidewalk/other-ground/terrain/tree/parking/stairs
        dsm_max = args.nearest_dsm_neighbor
        TRUSTED_LABELS = {1, 2, 3, 4, 7, 11, 12}  # road, sidewalk, parking, other-ground, terrain, tree, stairs
        far_from_dsm = dsm_d > dsm_max
        untrusted = ~np.isin(scan_labels, list(TRUSTED_LABELS))
        reset_mask = far_from_dsm & untrusted & (scan_labels != 0)
        if reset_mask.any():
            print(f"\nDSM proximity filter (>{dsm_max:.1f} m):")
            print(f"  {reset_mask.sum():,} points reset to unlabeled")
            scan_labels[reset_mask] = 0

        # Re-label unlabeled points above parking as vehicle
        unlabeled_mask = scan_labels == 0
        parking_vehicle = 0
        if spatial_idx.get("parking", (None, []))[0] is not None:
            park_tree, park_geoms = spatial_idx["parking"]
            for i in np.where(unlabeled_mask)[0]:
                pt = Point(sx[i], sy[i])
                if any(park_geoms[j].contains(pt)
                       for j in park_tree.query(pt)):
                    scan_labels[i] = 10
                    parking_vehicle += 1
            if parking_vehicle:
                print(f"  {parking_vehicle:,} unlabeled above parking → vehicle")

        print("\nDistribution:")
        for lid in sorted(LABEL_NAMES):
            c = (scan_labels == lid).sum()
            if c:
                print(f"  {lid:2d} {LABEL_NAMES[lid]:15s}: "
                      f"{c:>10,} ({100*c/len(scan_labels):5.1f}%)")

        # Semantic change detection (kernel-based)
        sem_change_mask = None
        if args.kernel_radius > 0:
            print(f"\nSemantic change detection "
                  f"(kernel_radius={args.kernel_radius:.2f} m)...")
            sem_change_mask = detect_semantic_changes(
                sx, sy, scan_labels, args.kernel_radius)

        # FPFH computation
        fpfh_colors = None
        if args.fpfh:
            target_pts = scan_pts_utm
            if args.show_sem_change_only and sem_change_mask is not None:
                target_pts = scan_pts_utm[sem_change_mask]
            print(f"\nComputing FPFH (radius={args.fpfh_radius:.2f} m, "
                  f"{len(target_pts):,} points)...")
            fpfh_desc = compute_fpfh(target_pts, args.fpfh_radius)
            fpfh_colors = fpfh_to_rgb(fpfh_desc)
            print(f"  FPFH → RGB via PCA complete")

    # ══════════════════════════════════════════════════════════════════
    # 8. Save per-scan labels (per robot)
    # ══════════════════════════════════════════════════════════════════
    if args.save_labels and scan_labels is not None and robot_data:
        print("\n" + "=" * 60)
        print("Saving per-scan labels")
        print("=" * 60)

        labeled_tree = KDTree(scan_pts_utm)
        max_nn = args.nn_max_dist
        offset = 0  # track position in combined scan_labels

        for rd in robot_data:
            rname = rd["robot"]
            out_dir = os.path.join(args.data_dir, args.env, rname,
                                   "groundtruth_labels")
            os.makedirs(out_dir, exist_ok=True)
            print(f"\n  {rname} → {out_dir}")

            saved = 0
            for count, idx in enumerate(rd["indices"]):
                pose = rd["poses"][idx]
                rot = Rotation.from_quat(pose[4:8])
                raw = np.fromfile(
                    os.path.join(rd["scan_bin_dir"], rd["scan_files"][idx]),
                    dtype=np.float32).reshape(-1, 4)
                pts = rot.apply(raw[:, :3].astype(np.float64)) + pose[1:4]
                pts[:, 2] += rd["z_shifts"][count]

                d, ix = labeled_tree.query(pts)
                lbl = scan_labels[ix].copy()
                lbl[d > max_nn] = 0

                lbl.astype(np.uint32).tofile(
                    os.path.join(out_dir, f"{idx:010d}.label"))
                saved += 1
                if saved % 100 == 0 or saved == len(rd["indices"]):
                    print(f"    [{saved}/{len(rd['indices'])}]")

            print(f"  {rname}: {saved} labels saved")

    # ══════════════════════════════════════════════════════════════════
    # 9. Visualisation
    # ══════════════════════════════════════════════════════════════════
    print("\nPreparing visualisation...")
    x_off, y_off = x.mean(), y.mean()
    x -= x_off; y -= y_off

    plotter = pv.Plotter()

    # DEM
    if args.show_dem:
        for gi, grid in enumerate(dem_grids):
            pts = grid.points.copy()
            pts[:, 0] -= x_off; pts[:, 1] -= y_off
            g = pv.StructuredGrid()
            g.dimensions = grid.dimensions; g.points = pts
            if args.dem_color_by_height:
                g["Elevation"] = pts[:, 2].copy()
                plotter.add_mesh(g, scalars="Elevation", cmap="terrain",
                                 opacity=0.6, show_edges=False,
                                 scalar_bar_args={"title": "Elevation (m)"})
            else:
                plotter.add_mesh(g, color=[.4, .35, .25], opacity=0.6,
                                 show_edges=False, label=f"DEM {gi}")

    # OSM flat polygons
    if args.show_osm:
        gz_base = np.median(z[cls == 2]) if (cls == 2).any() else np.median(z)
        oz = gz_base + args.osm_z_offset
        for mesh, rgba, _ in osm_meshes:
            pts = mesh.points.copy()
            pts[:, 0] -= x_off; pts[:, 1] -= y_off; pts[:, 2] = oz
            md = mesh.copy(); md.points = pts
            rgb_f = [c / 255 for c in rgba[:3]]
            opa = rgba[3] / 255 if len(rgba) > 3 else 0.7
            plotter.add_mesh(md, color=rgb_f, opacity=opa,
                             show_edges=False)

    # DSM
    if args.show_dsm:
        cloud = pv.PolyData(np.column_stack([x, y, z]))
        if args.dsm_color_by_height:
            cloud["Elevation"] = z.copy()
            plotter.add_mesh(cloud, scalars="Elevation", cmap="terrain",
                             point_size=args.point_size,
                             render_points_as_spheres=False,
                             scalar_bar_args={"title": "DSM Elevation (m)"})
        else:
            cols = np.full((len(x), 3), 180, dtype=np.uint8)
            for cid, rgb in CLASS_COLORS.items():
                m = cls == cid
                if m.any():
                    cols[m] = rgb
            cloud["RGB"] = cols
            plotter.add_mesh(cloud, scalars="RGB", rgb=True,
                             point_size=args.point_size,
                             render_points_as_spheres=False)

    # Scans
    vis_labels = scan_labels

    if scan_pts_utm is not None and args.show_scans:
        sd = scan_pts_utm.copy()
        sd[:, 0] -= x_off; sd[:, 1] -= y_off

        if args.show_sem_change_only and sem_change_mask is not None:
            sd = sd[sem_change_mask]
            vis_labels = vis_labels[sem_change_mask] if vis_labels is not None else None
            print(f"  Sem-change-only view: {sem_change_mask.sum():,} / "
                  f"{len(sem_change_mask):,} points")

        sc = pv.PolyData(sd)
        if args.fpfh and fpfh_colors is not None:
            sc["RGB"] = fpfh_colors
            plotter.add_mesh(sc, scalars="RGB", rgb=True,
                             point_size=args.scan_point_size,
                             render_points_as_spheres=False,
                             label="FPFH (PCA→RGB)")
        elif vis_labels is not None and args.label:
            cols = np.full((len(vis_labels), 3), 128, dtype=np.uint8)
            for lid, rgb in LABEL_COLORS.items():
                m = vis_labels == lid
                if m.any():
                    cols[m] = rgb
            sc["RGB"] = cols
            plotter.add_mesh(sc, scalars="RGB", rgb=True,
                             point_size=args.scan_point_size,
                             render_points_as_spheres=False,
                             label="Labelled Scans")
        else:
            plotter.add_mesh(sc, color=[1, .3, 0],
                             point_size=args.scan_point_size,
                             render_points_as_spheres=False, label="Scans")

    # Legend
    leg = []
    if args.show_osm:
        for cat in ("buildings", "roads", "sidewalks", "stairs",
                     "grasslands", "forests", "water"):
            if any(c == cat for _, _, c in osm_meshes):
                leg.append([f"OSM {cat}",
                            [c/255 for c in OSM_COLORS[cat][:3]]])
    if args.show_dem and dem_grids:
        leg.append(["DEM", [.4, .35, .25]])
    if scan_pts_utm is not None and args.show_scans:
        if args.fpfh and fpfh_colors is not None:
            leg.append(["FPFH (PCA→RGB)", [0.5, 0.5, 0.5]])
        elif vis_labels is not None and args.label:
            for lid in sorted(LABEL_NAMES):
                if (vis_labels == lid).any():
                    leg.append([LABEL_NAMES[lid],
                                [v/255 for v in LABEL_COLORS[lid]]])
        else:
            leg.append(["Scans", [1, .3, 0]])
    if leg:
        plotter.add_legend(leg, bcolor=(.9, .9, .9), face=None)

    # Title
    parts = []
    if args.show_dsm:
        parts.append(f"DSM ({len(las_paths)})")
    if args.show_dem:
        parts.append(f"DEM ({len(dem_grids)})")
    if args.show_osm and osm_features:
        parts.append(f"OSM ({sum(1 for p in osm_paths if os.path.isfile(p))})")
    if scan_pts_utm is not None:
        tag = " + Labels" if scan_labels is not None and args.label else ""
        parts.append(f"Scans ({scan_pts_utm.shape[0]:,}{tag})")
    title = " + ".join(parts) + "  [UTM 13N]" if parts else "UTM 13N"

    plotter.set_background("white")
    plotter.add_text(title, font_size=12, color="black")

    if args.save:
        plotter.show(screenshot=args.save)
        print(f"Saved screenshot to {args.save}")
    else:
        print("\nLaunching 3D viewer...")
        plotter.show()


if __name__ == "__main__":
    main()
