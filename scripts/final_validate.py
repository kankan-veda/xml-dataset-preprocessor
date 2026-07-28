#!/usr/bin/env python3
"""Step 5: Final validation of the split dataset.

Usage:
    python final_validate.py --dataset-dir <dataset> --log-dir <logs>

Dependencies: pip install Pillow
"""

import os
import argparse

try:
    from PIL import Image
except ImportError:
    Image = None


def count_files(directory):
    txts = set()
    imgs = set()
    for fname in os.listdir(directory):
        if fname.endswith(".txt"):
            txts.add(os.path.splitext(fname)[0])
        else:
            name, ext = os.path.splitext(fname)
            if ext.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                imgs.add(name)
    return txts, imgs


def verify_label_format(txt_path):
    issues = []
    try:
        with open(txt_path) as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    issues.append(f"  Empty line {lineno}")
                    continue
                parts = line.split()
                if len(parts) != 5:
                    issues.append(
                        f"  Line {lineno}: expected 5 columns, "
                        f"got {len(parts)}"
                    )
                    continue
                try:
                    cls = int(parts[0])
                except ValueError:
                    issues.append(
                        f"  Line {lineno}: class_id not int: {parts[0]}"
                    )
                    continue
                try:
                    vals = [float(v) for v in parts[1:]]
                    if any(v < 0 or v > 1 for v in vals):
                        issues.append(
                            f"  Line {lineno}: coord out of [0,1]: "
                            f"{parts[1:]}"
                        )
                except ValueError:
                    issues.append(
                        f"  Line {lineno}: non-numeric coord: "
                        f"{parts[1:]}"
                    )
    except Exception as e:
        issues.append(f"  Read error: {e}")
    return issues


def verify_image(img_path):
    if Image is None:
        return ["Pillow not installed, skipping image check"]
    try:
        with Image.open(img_path) as img:
            img.verify()
        return []
    except Exception as e:
        return [f"  Cannot open: {e}"]


def main():
    parser = argparse.ArgumentParser(
        description="Final dataset validation"
    )
    parser.add_argument("--dataset-dir", required=True,
                        help="Dataset directory (parent of train/val/test)")
    parser.add_argument("--log-dir", default="logs",
                        help="Log output directory")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    splits = ["train", "val", "test"]

    report_lines = []
    report_lines.append("=" * 56)
    report_lines.append("        Final Dataset Validation Report")
    report_lines.append("=" * 56)
    severity = "OK"

    # 1. Count consistency
    report_lines.append("\n[1] Count consistency per split")
    total_txt = 0
    total_img = 0
    for split in splits:
        labels_dir = os.path.join(args.dataset_dir, split, "labels")
        images_dir = os.path.join(args.dataset_dir, split, "images")
        if not os.path.isdir(labels_dir) or not os.path.isdir(images_dir):
            report_lines.append(f"  {split}: MISSING directory")
            severity = max(severity, "CRITICAL")
            continue

        n_txt = len([f for f in os.listdir(labels_dir)
                      if f.endswith(".txt")])
        img_files = [f for f in os.listdir(images_dir)
                     if os.path.splitext(f)[1].lower()
                     in (".jpg", ".jpeg", ".png", ".bmp", ".webp")]
        n_img = len(img_files)
        total_txt += n_txt
        total_img += n_img
        report_lines.append(f"  {split}: {n_txt} TXT, {n_img} images")
        diff = abs(n_txt - n_img)
        if diff == 0:
            report_lines.append(f"    OK")
        elif diff < 5:
            report_lines.append(f"    WARNING: {diff} file(s) mismatch")
            severity = max(severity, "WARNING")
        else:
            report_lines.append(f"    ERROR: {diff} file(s) mismatch")
            severity = max(severity, "ERROR")

    # 2. Total count
    report_lines.append("\n[2] Total sample count")
    report_lines.append(f"  TXT across all splits: {total_txt}")
    report_lines.append(f"  Images across all splits: {total_img}")
    if total_txt != total_img:
        report_lines.append(
            f"  ERROR: Total TXT ({total_txt}) != "
            f"Total images ({total_img})"
        )
        severity = max(severity, "CRITICAL")
    else:
        report_lines.append(f"  OK: Total matches ({total_txt})")

    # 3. Empty files
    report_lines.append("\n[3] Empty file check")
    empty_count = 0
    for split in splits:
        labels_dir = os.path.join(args.dataset_dir, split, "labels")
        if not os.path.isdir(labels_dir):
            continue
        for fname in os.listdir(labels_dir):
            if not fname.endswith(".txt"):
                continue
            path = os.path.join(labels_dir, fname)
            if os.path.getsize(path) == 0:
                report_lines.append(f"  EMPTY: {split}/labels/{fname}")
                empty_count += 1
    if empty_count == 0:
        report_lines.append("  OK: No empty files")
    else:
        report_lines.append(f"  {empty_count} empty TXT file(s) found")
        severity = max(severity, "ERROR")

    # 4. Format validation
    report_lines.append("\n[4] YOLO format validation")
    format_issues = 0
    for split in splits:
        labels_dir = os.path.join(args.dataset_dir, split, "labels")
        if not os.path.isdir(labels_dir):
            continue
        for fname in sorted(os.listdir(labels_dir)):
            if not fname.endswith(".txt"):
                continue
            path = os.path.join(labels_dir, fname)
            issues = verify_label_format(path)
            if issues:
                report_lines.append(f"  {split}/labels/{fname}:")
                for iss in issues:
                    report_lines.append(f"    {iss}")
                format_issues += len(issues)
    if format_issues == 0:
        report_lines.append("  OK: All files have valid YOLO format")
    else:
        report_lines.append(f"  {format_issues} format issue(s) found")
        severity = max(severity, "ERROR")

    # 5. Image integrity
    report_lines.append("\n[5] Image integrity check (PIL)")
    bad_images = 0
    for split in splits:
        images_dir = os.path.join(args.dataset_dir, split, "images")
        if not os.path.isdir(images_dir):
            continue
        for fname in sorted(os.listdir(images_dir)):
            name, ext = os.path.splitext(fname)
            if ext.lower() not in (".jpg", ".jpeg", ".png",
                                   ".bmp", ".webp"):
                continue
            path = os.path.join(images_dir, fname)
            issues = verify_image(path)
            if issues:
                report_lines.append(f"  {split}/images/{fname}:")
                for iss in issues:
                    report_lines.append(f"    {iss}")
                bad_images += 1
    if bad_images == 0:
        report_lines.append("  OK: All images are valid")
    else:
        report_lines.append(f"  {bad_images} corrupted/bad image(s)")
        severity = max(severity, "ERROR")

    # Final verdict
    report_lines.append("\n" + "=" * 56)
    report_lines.append(f"  Verdict: {severity}")
    report_lines.append("=" * 56)

    report_text = "\n".join(report_lines)
    print(report_text)

    report_path = os.path.join(args.log_dir, "validation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()