#!/usr/bin/env python3
"""Step 3A: Statistics + generate filter_plan.yaml. Does NOT delete anything.

Usage:
    python stats_and_filter.py --labels-dir <labels> --img-dir <images> \
        --output <filter_plan.yaml>

Output:
  1. Terminal: formatted statistics for Codex to read
  2. filter_plan.yaml: editable config for filtering
"""

import os
import argparse

try:
    import yaml
except ImportError:
    yaml = None


def collect_stats(labels_dir):
    stats = {
        "total_samples": 0,
        "total_objects": 0,
        "zero_object_samples": 0,
        "single_object_samples": 0,
        "zero_object_files": [],
        "single_object_files": [],
        "class_distribution": {},
        "object_count_distribution":
            {"0": 0, "1": 0, "2-3": 0, "4-7": 0, "8+": 0},
    }

    for fname in sorted(os.listdir(labels_dir)):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(labels_dir, fname)
        with open(path) as f:
            lines = [line.strip() for line in f if line.strip()]
        count = len(lines)
        stats["total_samples"] += 1
        stats["total_objects"] += count

        # Bucket
        if count == 0:
            stats["object_count_distribution"]["0"] += 1
            stats["zero_object_files"].append(fname)
            stats["zero_object_samples"] += 1
        elif count == 1:
            stats["object_count_distribution"]["1"] += 1
            stats["single_object_files"].append(fname)
            stats["single_object_samples"] += 1
        elif count <= 3:
            stats["object_count_distribution"]["2-3"] += 1
        elif count <= 7:
            stats["object_count_distribution"]["4-7"] += 1
        else:
            stats["object_count_distribution"]["8+"] += 1

        # Per-class counts
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 1:
                cls = parts[0]
                stats["class_distribution"][cls] = \
                    stats["class_distribution"].get(cls, 0) + 1

    return stats


def print_stats(stats):
    print("\n" + "=" * 48)
    print("          Dataset Statistics")
    print("=" * 48)
    print(f"  Total samples:        {stats['total_samples']}")
    print(f"  Total objects:        {stats['total_objects']}")
    print(f"  Zero-object samples:  {stats['zero_object_samples']}")
    print(f"  Single-object samples: {stats['single_object_samples']}")
    print("-" * 48)
    print("  Class distribution:")
    total = sum(stats["class_distribution"].values()) or 1
    for cls, cnt in sorted(stats["class_distribution"].items(),
                           key=lambda x: -x[1]):
        pct = cnt / total * 100
        print(f"    {cls:<20s} {cnt:>6d} ({pct:5.1f}%)")
    print("-" * 48)
    print("  Objects per sample:")
    for bucket, cnt in stats["object_count_distribution"].items():
        pct = cnt / max(stats["total_samples"], 1) * 100
        bar = "█" * int(cnt / max(stats["total_samples"], 1) * 40)
        print(f"    {bucket:>4s} targets: {cnt:>5d} ({pct:5.1f}%)  {bar}")
    print("=" * 48)

    if stats["zero_object_files"]:
        files = stats["zero_object_files"]
        print(f"\n  Zero-object files ({len(files)}):")
        for fn in files[:10]:
            print(f"    - {fn}")
        if len(files) > 10:
            print(f"    ... and {len(files) - 10} more")
    if stats["single_object_files"]:
        files = stats["single_object_files"]
        print(f"\n  Single-object files ({len(files)}):")
        for fn in files[:10]:
            print(f"    - {fn}")
        if len(files) > 10:
            print(f"    ... and {len(files) - 10} more")
    print()


def write_yaml(stats, output_path):
    if yaml is None:
        print("ERROR: pyyaml not installed. Run: pip install pyyaml")
        return

    plan = {
        "summary": {
            "total_samples": stats["total_samples"],
            "total_objects": stats["total_objects"],
            "zero_object_samples": stats["zero_object_samples"],
            "single_object_samples": stats["single_object_samples"],
            "zero_object_files": stats["zero_object_files"],
            "single_object_files": stats["single_object_files"],
        },
        "class_distribution": stats["class_distribution"],
        "object_count_distribution":
            stats["object_count_distribution"],
        "filter": {
            "remove_zero_object": False,
            "remove_single_object": False,
            "remove_below_threshold": 0,
            "also_remove_images": True,
        }
    }
    # Truncate long file lists for readability
    MAX_LIST = 20
    for key in ("zero_object_files", "single_object_files"):
        files = plan["summary"][key]
        if len(files) > MAX_LIST:
            plan["summary"][key] = (
                files[:MAX_LIST]
                + [f"... and {len(files) - MAX_LIST} more"]
            )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Filter plan generated by stats_and_filter.py\n")
        f.write("# Edit the 'filter' section below, then "
                "run apply_filter.py --plan <this file>\n\n")
        yaml.dump(plan, f, default_flow_style=False,
                  allow_unicode=True, sort_keys=False)
    print(f"filter_plan.yaml written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate dataset statistics and filter plan"
    )
    parser.add_argument("--labels-dir", required=True,
                        help="Directory containing TXT files")
    parser.add_argument("--img-dir", required=True,
                        help="Directory containing image files")
    parser.add_argument("--output", required=True,
                        help="Output filter_plan.yaml path")
    args = parser.parse_args()

    stats = collect_stats(args.labels_dir)
    print_stats(stats)
    write_yaml(stats, args.output)
    print("Done. Edit the filter section in the YAML file, "
          "then run apply_filter.py.")


if __name__ == "__main__":
    main()