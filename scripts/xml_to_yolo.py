#!/usr/bin/env python3
"""Step 1: Convert PASCAL VOC XML annotations to YOLO format TXT.

Usage:
    python xml_to_yolo.py --xml-dir <xmls> --img-dir <images> \
        --output-dir <labels> --classes-output <classes.txt>

Dependencies: pip install Pillow lxml
"""

import os
import sys
import argparse
from xml.etree import ElementTree as ET

try:
    from PIL import Image
except ImportError:
    Image = None


# ── Size extraction ─────────────────────────────────────

def get_size_from_xml(xml_path):
    """Read image size from <size> tag. Returns (w, h) or None."""
    tree = ET.parse(xml_path)
    size = tree.getroot().find("size")
    if size is None:
        return None
    w_el = size.find("width")
    h_el = size.find("height")
    if w_el is None or h_el is None:
        return None
    w, h = int(w_el.text), int(h_el.text)
    return None if w == 0 or h == 0 else (w, h)


def get_size_from_pil(img_dir, stem):
    """Read image size via PIL. Returns (w, h) or None."""
    if Image is None:
        raise RuntimeError("Pillow not installed. Run: pip install Pillow")
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        path = os.path.join(img_dir, stem + ext)
        if os.path.exists(path):
            with Image.open(path) as img:
                return img.size  # (width, height)
    return None


def get_image_size(xml_path, img_dir, log_fp):
    """Priority: XML -> PIL -> None (logged)."""
    wh = get_size_from_xml(xml_path)
    if wh is not None:
        return wh
    stem = os.path.splitext(os.path.basename(xml_path))[0]
    wh = get_size_from_pil(img_dir, stem)
    if wh is not None:
        return wh
    log_fp.write(
        f"{xml_path}: size missing, image not found at "
        f"{img_dir}\\{stem}.*, skipped\n"
    )
    return None


# ── Class collection ────────────────────────────────────

def collect_classes(xml_dir):
    """Scan all XMLs, return sorted list of unique class names."""
    classes = set()
    for fname in os.listdir(xml_dir):
        if not fname.lower().endswith(".xml"):
            continue
        tree = ET.parse(os.path.join(xml_dir, fname))
        for obj in tree.getroot().findall("object"):
            name = obj.find("name")
            if name is not None and name.text:
                classes.add(name.text.strip())
    return sorted(classes)


# ── XML -> TXT conversion ───────────────────────────────

def convert_one_xml(xml_path, img_dir, class_to_id, log_fp):
    """Convert single XML to list of YOLO lines. Returns None on failure."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    wh = get_image_size(xml_path, img_dir, log_fp)
    if wh is None:
        return None
    width, height = wh
    lines = []
    for obj in root.findall("object"):
        name_el = obj.find("name")
        if name_el is None or not name_el.text:
            continue
        name = name_el.text.strip()
        if name not in class_to_id:
            continue
        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        x_center = (xmin + xmax) / 2.0 / width
        y_center = (ymin + ymax) / 2.0 / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height

        # Clamp to [0, 1] to prevent overflow
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        w = max(0.0, min(1.0, w))
        h = max(0.0, min(1.0, h))

        lines.append(
            f"{class_to_id[name]} {x_center:.6f} "
            f"{y_center:.6f} {w:.6f} {h:.6f}"
        )
    return lines


# ── Main ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert VOC XML to YOLO TXT"
    )
    parser.add_argument("--xml-dir", required=True,
                        help="Directory containing XML files")
    parser.add_argument("--img-dir", required=True,
                        help="Directory containing image files")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for TXT files")
    parser.add_argument("--classes-output", required=True,
                        help="Output path for classes.txt")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Collect classes
    print("Scanning XMLs to collect class names...")
    classes = collect_classes(args.xml_dir)
    print(f"  Found {len(classes)} classes: {classes}")
    with open(args.classes_output, "w", encoding="utf-8") as f:
        for c in classes:
            f.write(c + "\n")
    class_to_id = {name: idx for idx, name in enumerate(classes)}

    # Convert
    print("Converting XML to YOLO format...")
    converted = 0
    skipped = 0
    errors = 0

    with open("logs/problematic.txt", "w", encoding="utf-8") as log_fp:
        for fname in os.listdir(args.xml_dir):
            if not fname.lower().endswith(".xml"):
                continue
            xml_path = os.path.join(args.xml_dir, fname)
            stem = os.path.splitext(fname)[0]
            try:
                lines = convert_one_xml(xml_path, args.img_dir,
                                        class_to_id, log_fp)
            except Exception as e:
                log_fp.write(f"{xml_path}: error {e}\n")
                errors += 1
                continue
            if lines is None:
                skipped += 1
                continue
            out_path = os.path.join(args.output_dir, stem + ".txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n" if lines else "")
            converted += 1

    # Summary
    print(f"  Converted:  {converted}")
    print(f"  Skipped:    {skipped} (size missing + image not found)")
    print(f"  Errors:     {errors}")
    print(f"  Total XMLs: {converted + skipped + errors}")
    print(f"\nclasses.txt written to: {args.classes_output}")
    print(f"Problematic samples logged to: logs/problematic.txt")

    # Validate output format
    print("\nValidating output format...")
    invalid = 0
    for fname in os.listdir(args.output_dir):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(args.output_dir, fname)
        with open(path) as f:
            for lineno, line in enumerate(f, 1):
                parts = line.strip().split()
                if len(parts) != 5:
                    print(f"  [INVALID] {fname}:{lineno} "
                          f"expected 5 columns, got {len(parts)}")
                    invalid += 1
                    continue
                try:
                    vals = [float(v) for v in parts[1:]]
                    if any(v < 0 or v > 1 for v in vals):
                        print(f"  [INVALID] {fname}:{lineno} "
                              f"coordinate out of [0,1] range")
                        invalid += 1
                except ValueError:
                    print(f"  [INVALID] {fname}:{lineno} "
                          f"non-numeric value")
                    invalid += 1
    if invalid == 0:
        print("  All TXT files are valid.")
    else:
        print(f"  {invalid} invalid lines found. "
              f"Check logs/conversion_check.txt for details.")
    print("Done.")


if __name__ == "__main__":
    main()