#!/usr/bin/env python3
"""
Analyze DCL-SLAM inter-robot loop closure logs from /tmp/dcl_output.

Parses the glog binary log files to extract statistics at each stage
of the loop closure pipeline:
  1. Descriptor matching (LidarIris)
  2. RANSAC inlier check
  3. ICP fitness check
  4. PCM consistency check

Usage:
  python3 analyze_loop_closures.py [--log-dir /tmp/dcl_output]
  python3 analyze_loop_closures.py [--log-dir /tmp/dcl_output] --run 2
      (select a specific run by index, 0=oldest, -1=latest)
"""

import argparse
import collections
import glob as globmod
import re
import sys
from pathlib import Path


ROBOT_NAMES = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}


def read_binary_log(path: Path) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def parse_descriptor_entries(data: bytes, robot_id: int):
    """Parse LidarIris descriptor matching log entries.

    Robot IDs in the "btn" field are raw int8_t bytes (e.g. \\x00 for robot 0,
    \\x01 for robot 1), NOT ASCII digits. The keyframe indices are ASCII ints.
    """
    # Successful matches: "Iris Inter Loop<id>] btn \x01-74 and \x02-78. Dis: 0.39. Bias:178"
    loop_pattern = (
        rb'Iris Inter Loop<(\d)>\] btn '
        rb'(.)-(\d+) and (.)-(\d+)\. Dis: ([\d.]+)\. Bias:(\d+)'
    )
    # Failed matches: same format but "Not loop", distance may be 1e+07 (sentinel) or real
    not_loop_pattern = (
        rb'Iris Inter Not loop<(\d)>\] btn '
        rb'(.)-(\d+) and (.)-(\d+)\. Dis: ([\d.e+]+)\. Bias:(\d+)'
    )

    matches = []
    for m in re.finditer(loop_pattern, data):
        matches.append({
            "type": "match",
            "robot_id": int(m.group(1)),
            "src_robot": m.group(2)[0],  # raw byte value = robot id
            "src_key": int(m.group(3)),
            "tgt_robot": m.group(4)[0],
            "tgt_key": int(m.group(5)),
            "distance": float(m.group(6)),
            "bias": int(m.group(7)),
        })

    rejects = []
    for m in re.finditer(not_loop_pattern, data):
        try:
            dist = float(m.group(6).rstrip(b'.'))
        except ValueError:
            continue
        rejects.append({
            "type": "reject",
            "robot_id": int(m.group(1)),
            "src_robot": m.group(2)[0],
            "src_key": int(m.group(3)),
            "tgt_robot": m.group(4)[0],
            "tgt_key": int(m.group(5)),
            "distance": dist,
            "bias": int(m.group(7)),
        })

    return matches, rejects


def parse_icp_ransac_entries(data: bytes):
    """Parse ICP and RANSAC verification log entries."""
    # ICP success: "[InterLoop<0>] inlier (0.95 < 0.05) fitness (0.035 < 0.4). Add."
    add_pattern = (
        rb'InterLoop<(\d)>\] inlier \(([\d.]+) < ([\d.]+)\) '
        rb'fitness \(([\d.]+) < ([\d.]+)\)\. Add\.'
    )
    # ICP failure: "[InterLoop<0>] ICP failed (58.27 > 0.4). Reject."
    icp_fail_pattern = (
        rb'InterLoop<(\d)>\] ICP failed \(([\d.]+) > ([\d.]+)\)\. Reject\.'
    )
    # RANSAC failure: "[InterLoop<0>] RANSAC failed (0.047 < 0.05). Reject."
    ransac_fail_pattern = (
        rb'InterLoop<(\d)>\] RANSAC failed \(([\d.]+) < ([\d.]+)\)\. Reject\.'
    )

    icp_successes = []
    for m in re.finditer(add_pattern, data):
        icp_successes.append({
            "robot_id": int(m.group(1)),
            "inlier_ratio": float(m.group(2)),
            "ransac_thresh": float(m.group(3)),
            "fitness": float(m.group(4)),
            "fitness_thresh": float(m.group(5)),
        })

    icp_failures = []
    for m in re.finditer(icp_fail_pattern, data):
        icp_failures.append({
            "robot_id": int(m.group(1)),
            "fitness": float(m.group(2)),
            "fitness_thresh": float(m.group(3)),
        })

    ransac_failures = []
    for m in re.finditer(ransac_fail_pattern, data):
        ransac_failures.append({
            "robot_id": int(m.group(1)),
            "inlier_ratio": float(m.group(2)),
            "ransac_thresh": float(m.group(3)),
        })

    return icp_successes, icp_failures, ransac_failures


def parse_consistent_loops(path: Path) -> int:
    """Count lines in a consistent_loop_closures file."""
    if not path.exists():
        return 0
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def histogram_buckets(values, edges):
    """Create a histogram from values using the given bin edges."""
    buckets = collections.OrderedDict()
    for i in range(len(edges) - 1):
        label = f"{edges[i]:.2f}-{edges[i+1]:.2f}"
        buckets[label] = 0
    overflow_label = f">{edges[-1]:.2f}"
    buckets[overflow_label] = 0

    for v in values:
        placed = False
        for i in range(len(edges) - 1):
            if edges[i] <= v < edges[i + 1]:
                label = f"{edges[i]:.2f}-{edges[i+1]:.2f}"
                buckets[label] += 1
                placed = True
                break
        if not placed:
            buckets[overflow_label] += 1

    return buckets


def bar(count, max_count, width=40):
    """Simple text bar chart."""
    if max_count == 0:
        return ""
    n = int(count / max_count * width)
    return "█" * n + "░" * (width - n)


def print_section(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def discover_runs(log_dir: Path):
    """Discover all distinct runs in the log directory.

    glog creates files like:
      a_distributed_mapping.TheHomeMachine.user.log.INFO.20260406-083005.98822
    The timestamp (20260406-083005) identifies a run.  Returns a sorted list
    of (timestamp_str, {prefix: Path}) dicts, oldest first.
    """
    pattern = re.compile(
        r'^([a-z])_distributed_mapping\..+\.log\.INFO\.(\d{8}-\d{6})\.\d+$'
    )
    runs = collections.defaultdict(dict)  # ts -> {prefix: path}
    for p in log_dir.iterdir():
        if p.is_symlink():
            continue
        m = pattern.match(p.name)
        if m:
            prefix, ts = m.group(1), m.group(2)
            runs[ts][prefix] = p

    return sorted(runs.items(), key=lambda x: x[0])


def analyze_robot(robot_id: int, log_dir: Path, log_file: Path = None):
    """Analyze all log data for a single robot."""
    prefix = chr(ord('a') + robot_id)
    name = ROBOT_NAMES.get(robot_id, str(robot_id))

    if log_file is None:
        # Fallback: follow symlink
        log_file = log_dir / f"{prefix}_distributed_mapping.INFO"

    if not log_file.exists():
        return None

    data = read_binary_log(log_file)

    desc_matches, desc_rejects = parse_descriptor_entries(data, robot_id)
    icp_successes, icp_failures, ransac_failures = parse_icp_ransac_entries(data)
    pcm_count = parse_consistent_loops(log_dir / f"consistent_loop_closures_{prefix}.txt")

    # Separate sentinel (no valid candidate) from real distance rejects
    sentinel_rejects = [r for r in desc_rejects if r["distance"] >= 1e6]
    real_rejects = [r for r in desc_rejects if r["distance"] < 1e6]

    return {
        "robot_id": robot_id,
        "name": name,
        "prefix": prefix,
        "desc_matches": desc_matches,
        "desc_rejects_sentinel": sentinel_rejects,
        "desc_rejects_real": real_rejects,
        "icp_successes": icp_successes,
        "icp_failures": icp_failures,
        "ransac_failures": ransac_failures,
        "pcm_consistent": pcm_count,
    }


def print_robot_report(r):
    """Print detailed report for a single robot."""
    name = r["name"]
    total_desc = len(r["desc_matches"]) + len(r["desc_rejects_sentinel"]) + len(r["desc_rejects_real"])
    total_icp_attempts = len(r["icp_successes"]) + len(r["icp_failures"]) + len(r["ransac_failures"])

    print_section(f"Robot {name} (id={r['robot_id']}, prefix='{r['prefix']}')")

    # --- Descriptor stage ---
    print(f"\n  ┌─ DESCRIPTOR MATCHING (LidarIris)")
    print(f"  │  Total comparisons:     {total_desc}")
    print(f"  │  No valid candidate:    {len(r['desc_rejects_sentinel']):>6}  (sentinel 1e+07, no KNN neighbor from other robot)")
    print(f"  │  Rejected (dist >= th): {len(r['desc_rejects_real']):>6}")
    print(f"  │  Matched  (dist <  th): {len(r['desc_matches']):>6}")
    if total_desc > 0:
        match_rate = len(r["desc_matches"]) / total_desc * 100
        print(f"  │  Match rate:            {match_rate:>5.1f}%")

    # Descriptor distance distribution
    real_dists = [e["distance"] for e in r["desc_rejects_real"]]
    match_dists = [e["distance"] for e in r["desc_matches"]]

    if real_dists or match_dists:
        print(f"  │")
        print(f"  │  Distance distribution (threshold = 0.4):")
        all_dists = [(d, "reject") for d in real_dists] + [(d, "match") for d in match_dists]
        edges = [0.0, 0.10, 0.20, 0.30, 0.35, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50]
        max_bucket = 0

        reject_buckets = histogram_buckets(real_dists, edges) if real_dists else {}
        match_buckets = histogram_buckets(match_dists, edges) if match_dists else {}

        # Merge for display
        all_labels = list((reject_buckets or match_buckets).keys())
        for label in all_labels:
            rc = reject_buckets.get(label, 0)
            mc = match_buckets.get(label, 0)
            max_bucket = max(max_bucket, rc + mc)

        for label in all_labels:
            rc = reject_buckets.get(label, 0)
            mc = match_buckets.get(label, 0)
            total = rc + mc
            if total == 0:
                continue
            b = bar(total, max_bucket, 30)
            detail = ""
            if mc > 0 and rc > 0:
                detail = f" ({mc} match, {rc} reject)"
            elif mc > 0:
                detail = f" (match)"
            else:
                detail = f" (reject)"
            print(f"  │    {label:>11s}: {total:>4d} {b}{detail}")

    if match_dists:
        print(f"  │")
        print(f"  │  Match distance stats: min={min(match_dists):.4f}  mean={sum(match_dists)/len(match_dists):.4f}  max={max(match_dists):.4f}")

    if real_dists:
        print(f"  │  Reject distance stats: min={min(real_dists):.4f}  mean={sum(real_dists)/len(real_dists):.4f}  max={max(real_dists):.4f}")
        near_miss = [d for d in real_dists if d < 0.45]
        if near_miss:
            print(f"  │  Near-miss rejects (0.40-0.45): {len(near_miss)} -- could be recovered by relaxing threshold")

    # Robot pair breakdown
    pair_match_counts = collections.Counter()
    for e in r["desc_matches"]:
        pair = tuple(sorted([e["src_robot"], e["tgt_robot"]]))
        pair_match_counts[pair] += 1
    pair_reject_counts = collections.Counter()
    for e in r["desc_rejects_real"]:
        pair = tuple(sorted([e["src_robot"], e["tgt_robot"]]))
        pair_reject_counts[pair] += 1

    all_pairs = set(pair_match_counts.keys()) | set(pair_reject_counts.keys())
    if all_pairs:
        print(f"  │")
        print(f"  │  Per robot-pair breakdown:")
        for pair in sorted(all_pairs):
            mc = pair_match_counts.get(pair, 0)
            rc = pair_reject_counts.get(pair, 0)
            total = mc + rc
            rate = mc / total * 100 if total > 0 else 0
            pair_label = f"{ROBOT_NAMES.get(pair[0], str(pair[0]))}-{ROBOT_NAMES.get(pair[1], str(pair[1]))}"
            print(f"  │    {pair_label}: {mc} match / {rc} reject ({rate:.1f}% match rate)")

    print(f"  │")

    # --- ICP/RANSAC stage ---
    print(f"  ├─ ICP / RANSAC VERIFICATION")
    print(f"  │  Total attempts:        {total_icp_attempts}")
    print(f"  │  RANSAC failures:       {len(r['ransac_failures']):>6}")
    print(f"  │  ICP failures:          {len(r['icp_failures']):>6}")
    print(f"  │  Passed (Add):          {len(r['icp_successes']):>6}")
    if total_icp_attempts > 0:
        pass_rate = len(r["icp_successes"]) / total_icp_attempts * 100
        print(f"  │  Pass rate:             {pass_rate:>5.1f}%")

    # ICP failure distribution
    icp_fail_scores = [e["fitness"] for e in r["icp_failures"]]
    if icp_fail_scores:
        print(f"  │")
        print(f"  │  ICP failure fitness scores (threshold = {r['icp_failures'][0]['fitness_thresh']:.1f}):")
        icp_edges = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
        icp_buckets = histogram_buckets(icp_fail_scores, icp_edges)
        max_b = max(icp_buckets.values()) if icp_buckets else 1
        for label, count in icp_buckets.items():
            if count == 0:
                continue
            b = bar(count, max_b, 30)
            print(f"  │    {label:>11s}: {count:>4d} {b}")
        print(f"  │  Stats: min={min(icp_fail_scores):.2f}  median={sorted(icp_fail_scores)[len(icp_fail_scores)//2]:.2f}  max={max(icp_fail_scores):.2f}  mean={sum(icp_fail_scores)/len(icp_fail_scores):.2f}")

    # RANSAC failure distribution
    ransac_ratios = [e["inlier_ratio"] for e in r["ransac_failures"]]
    if ransac_ratios:
        thresh = r["ransac_failures"][0]["ransac_thresh"]
        print(f"  │")
        print(f"  │  RANSAC inlier ratios (threshold = {thresh}):")
        print(f"  │  Stats: min={min(ransac_ratios):.4f}  max={max(ransac_ratios):.4f}  mean={sum(ransac_ratios)/len(ransac_ratios):.4f}")

    # ICP success fitness distribution
    icp_ok_scores = [e["fitness"] for e in r["icp_successes"]]
    if icp_ok_scores:
        print(f"  │")
        print(f"  │  Passed ICP fitness scores:")
        print(f"  │  Stats: min={min(icp_ok_scores):.4f}  median={sorted(icp_ok_scores)[len(icp_ok_scores)//2]:.4f}  max={max(icp_ok_scores):.4f}")

    print(f"  │")

    # --- PCM stage ---
    print(f"  └─ PCM CONSISTENCY")
    print(f"     Input (ICP passed):    {len(r['icp_successes'])}")
    print(f"     Consistent loops:      {r['pcm_consistent']}")
    if len(r["icp_successes"]) > 0:
        pcm_rate = r["pcm_consistent"] / len(r["icp_successes"]) * 100
        print(f"     PCM pass rate:         {pcm_rate:.1f}%")


def print_summary_table(results):
    """Print a compact summary table across all robots."""
    print_section("PIPELINE SUMMARY")

    # Header
    robot_labels = [f"Robot {r['name']}" for r in results]
    header = f"{'Stage':<30s}" + "".join(f"{l:>14s}" for l in robot_labels)
    print(f"\n  {header}")
    print(f"  {'─' * (30 + 14 * len(results))}")

    def row(label, values, fmt=">14d"):
        cols = "".join(f"{v:{fmt[1:]}}" if isinstance(fmt, str) else f"{v:>14s}" for v in values)
        print(f"  {label:<30s}{cols}")

    # Descriptor
    row("Descriptor: no candidate", [len(r["desc_rejects_sentinel"]) for r in results])
    row("Descriptor: rejected", [len(r["desc_rejects_real"]) for r in results])
    row("Descriptor: matched", [len(r["desc_matches"]) for r in results])

    totals = [len(r["desc_matches"]) + len(r["desc_rejects_real"]) + len(r["desc_rejects_sentinel"]) for r in results]
    rates = []
    for i, r in enumerate(results):
        if totals[i] > 0:
            rates.append(f"{len(r['desc_matches']) / totals[i] * 100:.1f}%")
        else:
            rates.append("N/A")
    row("  -> match rate", rates, fmt=">14s")

    print(f"  {'─' * (30 + 14 * len(results))}")

    # ICP/RANSAC
    row("RANSAC failures", [len(r["ransac_failures"]) for r in results])
    row("ICP failures", [len(r["icp_failures"]) for r in results])
    row("ICP passed (Add)", [len(r["icp_successes"]) for r in results])

    icp_totals = [len(r["icp_successes"]) + len(r["icp_failures"]) + len(r["ransac_failures"]) for r in results]
    rates = []
    for i, r in enumerate(results):
        if icp_totals[i] > 0:
            rates.append(f"{len(r['icp_successes']) / icp_totals[i] * 100:.1f}%")
        else:
            rates.append("N/A")
    row("  -> pass rate", rates, fmt=">14s")

    print(f"  {'─' * (30 + 14 * len(results))}")

    # PCM
    row("PCM consistent", [r["pcm_consistent"] for r in results])
    rates = []
    for r in results:
        n = len(r["icp_successes"])
        if n > 0:
            rates.append(f"{r['pcm_consistent'] / n * 100:.1f}%")
        else:
            rates.append("N/A")
    row("  -> PCM pass rate", rates, fmt=">14s")

    print()


def print_cross_robot_analysis(results):
    """Analyze descriptor matching across all robot pairs."""
    print_section("CROSS-ROBOT DESCRIPTOR ANALYSIS")

    # Collect all pair data across all robots
    pair_data = collections.defaultdict(lambda: {"matches": 0, "rejects": 0, "match_dists": [], "reject_dists": []})

    for r in results:
        for e in r["desc_matches"]:
            pair = tuple(sorted([e["src_robot"], e["tgt_robot"]]))
            pair_data[pair]["matches"] += 1
            pair_data[pair]["match_dists"].append(e["distance"])
        for e in r["desc_rejects_real"]:
            pair = tuple(sorted([e["src_robot"], e["tgt_robot"]]))
            pair_data[pair]["rejects"] += 1
            pair_data[pair]["reject_dists"].append(e["distance"])

    for pair in sorted(pair_data.keys()):
        d = pair_data[pair]
        total = d["matches"] + d["rejects"]
        rate = d["matches"] / total * 100 if total > 0 else 0
        pname = f"{ROBOT_NAMES.get(pair[0], str(pair[0]))} <-> {ROBOT_NAMES.get(pair[1], str(pair[1]))}"

        print(f"\n  {pname}:")
        print(f"    Matches: {d['matches']}  |  Rejects: {d['rejects']}  |  Rate: {rate:.1f}%")
        if d["match_dists"]:
            md = d["match_dists"]
            print(f"    Match distances:  min={min(md):.4f}  mean={sum(md)/len(md):.4f}  max={max(md):.4f}")
        if d["reject_dists"]:
            rd = d["reject_dists"]
            print(f"    Reject distances: min={min(rd):.4f}  mean={sum(rd)/len(rd):.4f}  max={max(rd):.4f}")
            near = sum(1 for v in rd if v < 0.45)
            if near:
                print(f"    Near-miss (0.40-0.45): {near} rejects could potentially be recovered")

    print()


def print_recommendations(results):
    """Print actionable recommendations based on the analysis."""
    print_section("RECOMMENDATIONS")

    # Check if any robot has 0 descriptor matches
    for r in results:
        if len(r["desc_matches"]) == 0 and len(r["desc_rejects_real"]) > 0:
            dists = [e["distance"] for e in r["desc_rejects_real"]]
            print(f"\n  [!] Robot {r['name']} found ZERO descriptor matches.")
            print(f"      All {len(dists)} real rejects have distances {min(dists):.4f} - {max(dists):.4f}")
            print(f"      The current threshold is 0.4. Consider raising it to capture these.")

    # Check near-miss descriptor rejects
    all_near_miss = []
    for r in results:
        for e in r["desc_rejects_real"]:
            if e["distance"] < 0.45:
                all_near_miss.append(e["distance"])
    if all_near_miss:
        print(f"\n  [*] {len(all_near_miss)} total near-miss descriptor rejects (distance 0.40-0.45).")
        print(f"      Raising descriptor_distance_threshold from 0.4 to 0.45 would recover these.")
        print(f"      This may increase false positives -- ICP/PCM should filter bad ones.")

    # Check ICP failure rates
    for r in results:
        icp_total = len(r["icp_successes"]) + len(r["icp_failures"]) + len(r["ransac_failures"])
        if icp_total > 0 and len(r["icp_failures"]) / icp_total > 0.5:
            print(f"\n  [*] Robot {r['name']}: {len(r['icp_failures'])}/{icp_total} ({len(r['icp_failures'])/icp_total*100:.0f}%) ICP failures.")
            scores = [e["fitness"] for e in r["icp_failures"]]
            median = sorted(scores)[len(scores) // 2]
            print(f"      Median failure score: {median:.2f} (threshold: {r['icp_failures'][0]['fitness_thresh']:.1f})")
            if median > 5:
                print(f"      High median suggests descriptor is matching genuinely different places.")

    # Check RANSAC failures
    for r in results:
        if len(r["ransac_failures"]) > 10:
            ratios = [e["inlier_ratio"] for e in r["ransac_failures"]]
            print(f"\n  [*] Robot {r['name']}: {len(r['ransac_failures'])} RANSAC failures.")
            print(f"      Inlier ratios: {min(ratios):.4f} - {max(ratios):.4f} (threshold: {r['ransac_failures'][0]['ransac_thresh']})")

    # Check PCM rates
    for r in results:
        n = len(r["icp_successes"])
        if n > 0 and r["pcm_consistent"] / n < 0.5:
            print(f"\n  [*] Robot {r['name']}: PCM only kept {r['pcm_consistent']}/{n} ({r['pcm_consistent']/n*100:.0f}%) of ICP-passed loops.")
            print(f"      Low PCM rate suggests many ICP matches are spatially inconsistent (possible false positives).")

    print()


def main():
    parser = argparse.ArgumentParser(description="Analyze DCL-SLAM inter-robot loop closure logs")
    parser.add_argument("--log-dir", type=str, default="/tmp/dcl_output",
                        help="Directory containing DCL-SLAM log files (default: /tmp/dcl_output)")
    parser.add_argument("--run", type=int, default=None,
                        help="Select run by index (0=oldest, -1=latest). "
                             "If omitted, uses the largest (most complete) log per robot.")
    parser.add_argument("--list-runs", action="store_true",
                        help="List all available runs and exit")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"Error: log directory {log_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    runs = discover_runs(log_dir)

    if args.list_runs:
        print(f"Found {len(runs)} run(s) in {log_dir}:\n")
        for i, (ts, prefixes) in enumerate(runs):
            robots = ", ".join(sorted(prefixes.keys()))
            sizes = ", ".join(f"{p.name}: {p.stat().st_size/1024:.0f}KB" for p in prefixes.values())
            print(f"  [{i}] {ts}  robots: {robots}")
            for p in sorted(prefixes.values(), key=lambda x: x.name):
                print(f"        {p.name}  ({p.stat().st_size/1024:.0f} KB)")
        return

    # Select which log file to use per robot
    robot_logs = {}  # robot_id -> Path
    if args.run is not None:
        if not runs:
            print("No runs found!", file=sys.stderr)
            sys.exit(1)
        ts, prefixes = runs[args.run]
        print(f"Analyzing run {args.run} (timestamp: {ts})")
        for prefix, path in prefixes.items():
            robot_id = ord(prefix) - ord('a')
            robot_logs[robot_id] = path
    else:
        # Use the largest log file per robot (most complete run)
        all_files = collections.defaultdict(list)
        for ts, prefixes in runs:
            for prefix, path in prefixes.items():
                robot_id = ord(prefix) - ord('a')
                all_files[robot_id].append(path)
        for robot_id, paths in all_files.items():
            robot_logs[robot_id] = max(paths, key=lambda p: p.stat().st_size)

    if not robot_logs:
        # Fallback to symlinks
        for i in range(10):
            prefix = chr(ord('a') + i)
            link = log_dir / f"{prefix}_distributed_mapping.INFO"
            if link.exists():
                robot_logs[i] = link

    print(f"Analyzing DCL-SLAM logs in: {log_dir}")
    for robot_id, path in sorted(robot_logs.items()):
        print(f"  Robot {ROBOT_NAMES.get(robot_id, str(robot_id))}: {path.name} ({path.stat().st_size/1024:.0f} KB)")

    results = []
    for robot_id, path in sorted(robot_logs.items()):
        r = analyze_robot(robot_id, log_dir, log_file=path)
        if r is not None:
            results.append(r)

    if not results:
        print("No robot logs found!", file=sys.stderr)
        sys.exit(1)

    print(f"\nFound {len(results)} robot(s): {', '.join(r['name'] for r in results)}")

    # Summary table first
    print_summary_table(results)

    # Cross-robot pair analysis
    print_cross_robot_analysis(results)

    # Per-robot detailed reports
    for r in results:
        print_robot_report(r)

    # Recommendations
    print_recommendations(results)


if __name__ == "__main__":
    main()
