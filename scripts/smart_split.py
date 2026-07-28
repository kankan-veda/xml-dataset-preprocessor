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
    """Find matching image (case-insensitive)."""
    stem_lower = stem.lower()
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        for fname in os.listdir(img_dir):
            name, e = os.path.splitext(fname)
            if name.lower() == stem_lower and e.lower() == ext:
                return os.path.join(img_dir, fname), ext
    return None, None


def load_label_index(labels_dir):
    """Build per-sample info: file, class_counts, total objects."""
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
    """Return (total_deviation, per_class_report, sets, set_totals)."""
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

    # 每个子集的总目标数
    set_totals = {}
    for sp in ("train", "val", "test"):
        set_totals[sp] = sum(
            s["class_counts"].get(cls, 0)
            for s in sets[sp]
            for cls in s["class_counts"]
        )

    return total_dev, class_report, sets, set_totals


def optimize(assignment, samples, max_iters=100):
    """Iterative local search: swap samples to minimize deviation."""
    target = {"train": 0.7, "val": 0.2, "test": 0.1}
    total = len(samples)
    for _ in range(max_iters):
        current_dev, _, sets, _ = evaluate(assignment, samples)
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
    """Copy files to dataset/{train,val,test}/{images,labels}/."""
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


def print_report(assignment, samples, class_names=None):
    """打印两级报表：样本级分布 + 类别级分布"""
    target = {"train": 0.7, "val": 0.2, "test": 0.1}
    total_dev, class_report, sets, set_totals = evaluate(assignment, samples)
    total = len(samples)

    print("\n" + "=" * 66)
    print("              数据集划分报告")
    print("=" * 66)

    # ── Part 1: 样本数分布 ──
    print(f"\n  总样本数: {total}")
    print("-" * 66)
    print(f"  {'子集':<8s} {'样本数':>8s} {'比例':>8s} {'目标值':>8s} {'偏差':>8s}")
    print("-" * 66)
    for split in ("train", "val", "test"):
        cnt = len(sets[split])
        ratio = cnt / total if total else 0
        diff = ratio - target[split]
        bar = "█" * int(cnt / max(total, 1) * 30)
        print(f"  {split:<8s} {cnt:>8d} {ratio:>7.1%} "
              f"{target[split]:>7.0%} {diff:>+7.2%}  {bar}")
    print(f"\n  总偏差: {total_dev:.4f}")

    # ── Part 2: 每类别在三个子集中的分布 ──
    print("\n" + "=" * 66)
    print("  各类别在子集中的目标数分布")
    print("=" * 66)
    h  = f"  {'类别':<12s} {'总数':>6s} {'训练':>8s}"
    h += f" {'(占类)':>8s} {'(占集)':>8s}"
    h += f" {'验证':>8s} {'(占类)':>8s} {'(占集)':>8s}"
    h += f" {'测试':>8s} {'(占类)':>8s} {'(占集)':>8s}"
    print(h)
    print("-" * 66)

    for cls in sorted(class_report.keys()):
        c = class_report[cls]
        cls_total = c["train"] + c["val"] + c["test"]
        if cls_total == 0:
            continue

        t  = c["train"] / cls_total * 100 if cls_total else 0
        v  = c["val"]   / cls_total * 100 if cls_total else 0
        te = c["test"]  / cls_total * 100 if cls_total else 0

        tp  = c["train"] / set_totals["train"] * 100 if set_totals["train"] else 0
        vp  = c["val"]   / set_totals["val"]   * 100 if set_totals["val"]   else 0
        tep = c["test"]  / set_totals["test"]  * 100 if set_totals["test"]  else 0

        label = class_names[int(cls)] if class_names else cls

        print(f"  {label:<12s} {cls_total:>6d}"
              f" {c['train']:>8d} {t:>6.1f}% {tp:>6.1f}%"
              f" {c['val']:>8d} {v:>6.1f}% {vp:>6.1f}%"
              f" {c['test']:>8d} {te:>6.1f}% {tep:>6.1f}%")

    print("-" * 66)
    g_total = sum(set_totals.values())
    print(f"  {'合计':<12s} {g_total:>6d}"
          f" {set_totals['train']:>8d} {'100.0%':>8s} {'100.0%':>8s}"
          f" {set_totals['val']:>8d} {'100.0%':>8s} {'100.0%':>8s}"
          f" {set_totals['test']:>8d} {'100.0%':>8s} {'100.0%':>8s}")
    print("=" * 66)


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

    # 读取类别名称
    class_names = []
    with open(args.classes_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                class_names.append(line)
    print(f"  {len(class_names)} classes: {class_names}")

    ratios = {"train": 0.7, "val": 0.2, "test": 0.1}

    print("Initializing stratified assignment...")
    assignment = assign_initial(samples, ratios)

    print("Optimizing assignment...")
    assignment = optimize(assignment, samples, max_iters=100)

    # 打印两级报表（传类别名称）
    print_report(assignment, samples, class_names=class_names)

    print("Copying files...")
    copy_split(assignment, samples,
               args.labels_dir, args.img_dir, args.output_dir)
    print(f"  Files written to: {args.output_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
