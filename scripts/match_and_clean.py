#!/usr/bin/env python3
"""Step 2: Match TXT labels with images, remove orphan TXTs.

Usage:
    python match_and_clean.py --labels-dir <labels> --img-dir <images>

Removes any .txt that has no corresponding image file.
"""

import os
import argparse


def find_image(img_dir, stem):
    """Find first matching image (case-insensitive). Returns path or None."""
    stem_lower = stem.lower()
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        for fname in os.listdir(img_dir):
            name, e = os.path.splitext(fname)
            if name.lower() == stem_lower and e.lower() == ext:
                return os.path.join(img_dir, fname)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Match TXT with images, remove orphan TXTs"
    )
    parser.add_argument("--labels-dir", required=True,
                        help="Directory containing TXT files")
    parser.add_argument("--img-dir", required=True,
                        help="Directory containing image files")
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)

    txt_files = [f for f in os.listdir(args.labels_dir)
                 if f.endswith(".txt")]

    # Build set of image stems for reverse check
    img_stems = set()
    for fname in os.listdir(args.img_dir):
        name, ext = os.path.splitext(fname)
        if ext.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            img_stems.add(name.lower())

    removed = 0
    kept = 0
    deletion_log = []

    print(f"Scanning {len(txt_files)} TXT files against {args.img_dir}...")
    for fname in txt_files:
        stem = os.path.splitext(fname)[0]
        txt_path = os.path.join(args.labels_dir, fname)
        img_path = find_image(args.img_dir, stem)
        if img_path is None:
            os.remove(txt_path)
            removed += 1
            deletion_log.append(fname)
            print(f"  REMOVED {fname} (no matching image)")
        else:
            kept += 1

    # Reverse check: images without TXT
    orphan_images = []
    for fname in os.listdir(args.img_dir):
        name, ext = os.path.splitext(fname)
        if ext.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            continue
        txt_path = os.path.join(args.labels_dir, name + ".txt")
        if not os.path.exists(txt_path):
            orphan_images.append(fname)

    with open("logs/deletion_log.txt", "w", encoding="utf-8") as f:
        for fn in deletion_log:
            f.write(fn + "\n")

    print(f"\n  Kept:   {kept}")
    print(f"  Removed: {removed}")
    if orphan_images:
        print(f"  Images without TXT: {len(orphan_images)} "
              f"(left untouched)")
        for fn in orphan_images:
            print(f"    {fn}")
    print(f"Deletion log: logs/deletion_log.txt")
    print("Done.")


if __name__ == "__main__":
    main()