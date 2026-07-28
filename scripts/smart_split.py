#!/usr/bin/env python3
"""Step 4: Smart split dataset into train/val/test at 7:2:1 ratio.

Uses stratified sampling + iterative optimization.

Usage:
    python smart_split.py --labels-dir <cleaned> --img-dir <img> \
        --output-dir <dataset> --classes-file <classes.txt> [--seed 42]
"""

import os
import shutil
import random
import argparse
from collections import defaultdict


def find_image(img_dir, stem):
    stem_lower = stem.lower()
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        for fname in os.listdir(img_dir):
            name, e = os.path.splitext(fname)
            if name.lower() == stem_lower and e.lower() == ext:
                return os.path.join(img_dir, fname), ext
    return None, None


def load_label_index(labels_dir):
    samples = []
    for fname in sorted(os.listdir(labels_dir)):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(labels_dir, fname)
        with open(path) as f:
            lines = [line.strip() for line in f if line.strip()]
        if not lines:
            continue
        counts = defaultdict(int)
        for line in lines:
            cls = line.split()[0]
            counts[cls] += 1
        samples.append({
            "file": fname,
            "class_counts": dict(counts),
            "total": len(lines),
        })
    return samples


def assign_initial(samples, ratios):
    """Stratified initial assignment per class."""
    class_samples = defaultdict(list)
    for s in samples:
        for cls in s["class_counts"]:
            class_samples[cls].append(s)

    assignment = {s["file"]: None for s in samples}
    remaining = set(s["file"] for s in samples)

    for cls, cls_samps in sorted(class_samples.items(),
                                 key=lambda x: -len(x[1])):
        available = [s for s in cls_samps if s["file"] in remaining]
        random.shuffle(available)
        n = len(available)
        n_test = max(1, int(n * ratios["test"]))
        n_val = max(1, int(n * ratios["val"]))
        n_train = n - n_test - n_val

        for i, s in enumerate(available):
            if i < n_test:
                assignment[s["file"]] = "test"
            elif i < n_test + n_val:
                assignment[s["file"]] = "val"
            else:
                assignment[s["file"]] = "train"
            remaining.discard(s["file"])

    for s in samples:
        if assignment[s["file"]] is None:
            assignment[s["file"]] = "train"
    return assignment


def evaluate(assignment, samples):
    sets = {"train": [], "val": [], "test": []}
    for s in samples:
        sets[assignment[s["file"]]].append(s)
    total = len(samples)
    actual_ratios = {k: len(v) / total for k, v in sets.items()}
    target = {"train": 0.7, "val": 0.2, "test": 0.1}
    total_dev = sum(abs(actual_ratios[k] - target[k]) for k in target)

    all_classes = set()
    for s in samples:
        all_classes.update(s["class_counts"].keys())
    class_report = {}
    for cls in sorted(all_classes):
        counts = {}
        for split in ("train", "val", "test"):
            cnt = sum(
                s["class_counts"].get(cls, 0) for s in sets[split]
            )
            counts[split] = cnt
        class_report[cls] = counts
    return total_dev, class_report, sets


def optimize(assignment, samples, max_iters=100):
    target = {"train": 0.7, "val": 0.2, "test": 0.1}
    total = len(samples)
    for _ in range(max_iters):
        current_dev, _, sets = evaluate(assignment, samples)
        improved = False
        for s in samples:
            current_split = assignment[s["file"]]
            for new_split in ("train", "val", "test"):
                if new_split == current_split:
                    continue
                old_A = len(sets[current_split])
                old_B = len(sets[new_split])
                if total == 0:
                    continue
                old_dev = (
                    abs(old_A / total - target[current_split]) +
                    abs(old_B / total - target[new_split])
                )
                new_dev = (
                    abs((old_A - 1) / total - target[current_split]) +
                    abs((old_B + 1) / total - target[new_split])
                )
                if new_dev < old_dev - 0.001:
                    assignment[s["file"]] = new_split
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return assignment


def copy_split(assignment, samples,
               src_labels_dir, src_img_dir, output_dir):
    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(output_dir, split, "images"),
                    exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, "labels"),
                    exist_ok=True)
    for s in samples:
        split = assignment[s["file"]]
        stem = os.path.splitext(s["file"])[0]

        shutil.copy2(
            os.path.join(src_labels_dir, s["file"]),
            os.path.join(output_dir, split, "labels", s["file"])
        )
        img_path, ext = find_image(src_img_dir, stem)
        if img_path:
            shutil.copy2(
                img_path,
                os.path.join(output_dir, split, "images", stem + ext)
            )


def print_report(assignment, samples):
    total_dev, class_report, sets = evaluate(assignment, samples)
    total = len(samples)
    target = {"train": 0.7, "val": 0.2, "test": 0.1}

    print("\n" + "=" * 52)
    print("         Dataset Split Report")
    print("=" * 52)
    print(f"  Total samples: {total}")
    print("-" * 52)
    print(f"  {'Split':<10s} {'Count':>6s} {'Ratio':>8s} "
          f"{'Target':>8s} {'Diff':>8s}")
    print("-" * 52)
    for split in ("train", "val", "test"):
        cnt = len(sets[split])
        ratio = cnt / total if total else 0
        diff = ratio - target[split]
        print(f"  {split:<10s} {cnt:>6d} {ratio:>7.3f} "
              f"{target[split]:>7.1f} {diff:>+7.4f}")
    print(f"\n  Total deviation: {total_dev:.4f}")
    print("-" * 52)
    print(f"  {'Class':<15s} {'Train':>6s} {'Val':>6s} {'Test':>6s}")
    print("-" * 52)
    for cls in sorted(class_report.keys()):
        c = class_report[cls]
        print(f"  {cls:<15s} {c['train']:>6d} "
              f"{c['val']:>6d} {c['test']:>6d}")
    print("=" * 52)


def main():
    parser = argparse.ArgumentParser(
        description="Smart split dataset 7:2:1"
    )
    parser.add_argument("--labels-dir", required=True,
                        help="Labels directory (cleaned)")
    parser.add_argument("--img-dir", required=True,
                        help="Image directory")
    parser.add_argument("--output-dir", required=True,
                        help="Output dataset directory")
    parser.add_argument("--classes-file", required=True,
                        help="classes.txt path")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    print("Loading samples...")
    samples = load_label_index(args.labels_dir)
    print(f"  Loaded {len(samples)} samples")

    with open(args.classes_file) as f:
        classes = [line.strip() for line in f if line.strip()]
    print(f"  {len(classes)} classes")

    ratios = {"train": 0.7, "val": 0.2, "test": 0.1}

    print("Initializing stratified assignment...")
    assignment = assign_initial(samples, ratios)

    print("Optimizing assignment...")
    assignment = optimize(assignment, samples, max_iters=100)

    print_report(assignment, samples)

    print("Copying files...")
    copy_split(assignment, samples,
               args.labels_dir, args.img_dir, args.output_dir)
    print(f"  Files written to: {args.output_dir}")
    print("Done.")


if __name__ == "__main__":
    main()