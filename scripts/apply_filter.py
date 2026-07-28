#!/usr/bin/env python3
"""Step 3B: Read filter_plan.yaml and execute filtering.

Usage:
    python apply_filter.py --plan <filter_plan.yaml> \
        --labels-dir <labels> --img-dir <images> --output-dir <cleaned>

Dependencies: pip install pyyaml
"""

import os
import shutil
import argparse

try:
    import yaml
except ImportError:
    yaml = None


def find_image(img_dir, stem):
    stem_lower = stem.lower()
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        for fname in os.listdir(img_dir):
            name, e = os.path.splitext(fname)
            if name.lower() == stem_lower and e.lower() == ext:
                return os.path.join(img_dir, fname)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Filter dataset based on filter_plan.yaml"
    )
    parser.add_argument("--plan", required=True,
                        help="Path to filter_plan.yaml")
    parser.add_argument("--labels-dir", required=True,
                        help="Source TXT directory")
    parser.add_argument("--img-dir", required=True,
                        help="Source image directory")
    parser.add_argument("--output-dir", required=True,
                        help="Output cleaned directory")
    args = parser.parse_args()

    if yaml is None:
        print("ERROR: pyyaml not installed. Run: pip install pyyaml")
        return

    with open(args.plan, encoding="utf-8") as f:
        plan = yaml.safe_load(f)
    rules = plan.get("filter", {})

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Determine which files to remove
    to_remove = set()

    if rules.get("remove_zero_object"):
        for entry in plan["summary"].get("zero_object_files", []):
            if entry.startswith("... and "):
                continue
            to_remove.add(entry)
        print(f"  Rule: remove zero-object files -> "
              f"{len(to_remove)} so far")

    if rules.get("remove_single_object"):
        for entry in plan["summary"].get("single_object_files", []):
            if entry.startswith("... and "):
                continue
            to_remove.add(entry)
        print(f"  Rule: remove single-object files -> "
              f"{len(to_remove)} so far")

    threshold = rules.get("remove_below_threshold", 0)
    if threshold > 0:
        for fname in os.listdir(args.labels_dir):
            if not fname.endswith(".txt"):
                continue
            path = os.path.join(args.labels_dir, fname)
            with open(path) as f:
                count = sum(1 for line in f if line.strip())
            if count < threshold:
                to_remove.add(fname)
        print(f"  Rule: remove files with <{threshold} objects "
              f"-> {len(to_remove)} total candidates")

    also_images = rules.get("also_remove_images", True)

    if not to_remove:
        print("  No filter rules enabled. "
              "Copying all files to cleaned/.")

    removed_txt = 0
    removed_img = 0
    copied = 0

    for fname in os.listdir(args.labels_dir):
        if not fname.endswith(".txt"):
            continue
        src_txt = os.path.join(args.labels_dir, fname)
        stem = os.path.splitext(fname)[0]

        if fname in to_remove:
            removed_txt += 1
            if also_images:
                img_path = find_image(args.img_dir, stem)
                if img_path:
                    os.remove(img_path)
                    removed_img += 1
            continue

        # Copy TXT
        shutil.copy2(src_txt, os.path.join(args.output_dir, fname))
        # Copy image
        img_path = find_image(args.img_dir, stem)
        if img_path:
            _, ext = os.path.splitext(img_path)
            shutil.copy2(img_path,
                         os.path.join(args.output_dir, stem + ext))
        copied += 1

    print(f"\n  Files copied to cleaned/: {copied}")
    print(f"  TXT files removed:    {removed_txt}")
    print(f"  Images removed:       {removed_img}")

    with open("logs/filter_execution.log", "w", encoding="utf-8") as f:
        f.write(f"Copied: {copied}\n")
        f.write(f"TXT removed: {removed_txt}\n")
        f.write(f"Images removed: {removed_img}\n")
        for fn in sorted(to_remove):
            f.write(f"REMOVED: {fn}\n")
    print(f"Execution log: logs/filter_execution.log")
    print("Done.")


if __name__ == "__main__":
    main()