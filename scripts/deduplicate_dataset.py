#!/usr/bin/env python3
"""Step 2.5: Deduplicate image+label pairs by image content.

Finds byte-identical images (same content, possibly different filenames),
keeps the first pair (image + matching TXT label), and moves every other
pair into a quarantine directory. This prevents the same image from being
copied into both train and test splits.

Usage:
    python deduplicate_dataset.py --img-dir <images> --labels-dir <labels> \
        --output-dir <duplicates>

Add --dry-run to only report duplicates without moving anything.
"""

import argparse
import hashlib
import os
import shutil
from collections import defaultdict

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def file_sha256(path, chunk_size=1024 * 1024):
    """Return SHA-256 hex digest for a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def label_sha256(path):
    """Return SHA-256 for normalized TXT content (ignores blank lines)."""
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    normalized = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def index_files(directory, exts):
    """Return {lower_stem: (stem, path)} for files with matching extensions."""
    index = {}
    for fname in os.listdir(directory):
        stem, ext = os.path.splitext(fname)
        if ext.lower() not in exts:
            continue
        index[stem.lower()] = (stem, os.path.join(directory, fname))
    return index


def unique_dest(dest_dir, fname):
    """Return a destination path that will not overwrite an existing file."""
    candidate = os.path.join(dest_dir, fname)
    if not os.path.exists(candidate):
        return candidate
    stem, ext = os.path.splitext(fname)
    i = 1
    while True:
        candidate = os.path.join(dest_dir, f"{stem}_{i}{ext}")
        if not os.path.exists(candidate):
            return candidate
        i += 1


def main():
    parser = argparse.ArgumentParser(
        description="Remove duplicate image+label pairs before splitting"
    )
    parser.add_argument("--img-dir", required=True,
                        help="Image directory")
    parser.add_argument("--labels-dir", required=True,
                        help="YOLO TXT label directory")
    parser.add_argument("--output-dir", default="duplicates",
                        help="Directory for quarantined duplicate files")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report duplicates without moving files")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    img_dir = os.path.abspath(args.img_dir)
    labels_dir = os.path.abspath(args.labels_dir)
    if output_dir in (img_dir, labels_dir):
        parser.error("--output-dir must differ from --img-dir and --labels-dir")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    images = index_files(args.img_dir, IMAGE_EXTS)
    labels = index_files(args.labels_dir, (".txt",))

    print(f"Scanning {len(images)} images and {len(labels)} labels...")

    groups = defaultdict(list)
    for stem_lower, (stem, path) in sorted(images.items()):
        groups[file_sha256(path)].append((stem_lower, stem, path))

    dup_groups = 0
    dup_images = 0
    dup_labels = 0
    warnings = 0
    log_lines = []

    for digest, members in sorted(groups.items()):
        if len(members) < 2:
            continue
        dup_groups += 1
        keep_stem_lower, _, keep_path = members[0]
        keep_label = labels.get(keep_stem_lower)
        keep_label_path = keep_label[1] if keep_label else None
        keep_label_digest = (label_sha256(keep_label_path)
                             if keep_label_path else None)

        print(f"\nDuplicate group #{dup_groups} "
              f"({len(members)} images, hash {digest[:12]})")
        print(f"  KEEP   {os.path.basename(keep_path)}"
              + ("" if keep_label_path else "  (no label)"))

        for stem_lower, _, path in members[1:]:
            label = labels.get(stem_lower)
            label_path = label[1] if label else None
            label_digest = (label_sha256(label_path)
                            if label_path else None)

            dup_images += 1
            dest_img = unique_dest(args.output_dir, os.path.basename(path))
            if args.dry_run:
                print(f"  WOULD MOVE {os.path.basename(path)}")
                log_lines.append(
                    f"image\t{path}\t->\t{dest_img}\t(dry run)")
            else:
                shutil.move(path, dest_img)
                print(f"  MOVE   {os.path.basename(path)}")
                log_lines.append(f"image\t{path}\t->\t{dest_img}")

            if label_path:
                dup_labels += 1
                dest_label = unique_dest(
                    args.output_dir, os.path.basename(label_path))
                if args.dry_run:
                    print(f"  WOULD MOVE {os.path.basename(label_path)}")
                    log_lines.append(
                        f"label\t{label_path}\t->\t{dest_label}\t(dry run)")
                else:
                    shutil.move(label_path, dest_label)
                    print(f"  MOVE   {os.path.basename(label_path)}")
                    log_lines.append(f"label\t{label_path}\t->\t{dest_label}")

                if (keep_label_digest is not None
                        and label_digest != keep_label_digest):
                    warnings += 1
                    msg = (f"label content differs: "
                           f"{os.path.basename(keep_label_path)} vs "
                           f"{os.path.basename(label_path)}")
                    print(f"  WARNING {msg}")
                    log_lines.append(f"warning\t{msg}")
            else:
                print("  NOTE   duplicate image has no matching label")

    with open("logs/dedup_log.txt", "w", encoding="utf-8") as f:
        for line in log_lines:
            f.write(line + "\n")

    action = "would move" if args.dry_run else "moved"
    print("\n" + "=" * 48)
    print("  Duplicate image groups: " + str(dup_groups))
    print("  Duplicate images " + action + ": " + str(dup_images))
    print("  Duplicate labels " + action + ": " + str(dup_labels))
    print("  Label conflicts:        " + str(warnings))
    print("  Quarantine directory:   " + args.output_dir)
    print("  Log:                    logs/dedup_log.txt")
    if args.dry_run:
        print("\n  DRY RUN finished. No files were moved.")
    elif dup_groups == 0:
        print("\n  No duplicates found. Everything stays in place.")
    else:
        print("\n  Review the quarantine directory, then delete it "
              "when you are sure the moves are correct.")
    print("Done.")


if __name__ == "__main__":
    main()
