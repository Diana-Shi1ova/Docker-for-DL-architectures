#!/usr/bin/env python3
"""Split two class folders into train/val/test folders for PyTorch ImageFolder.

Expected source layout:
    butterfly_data/
        admiral/
            image001.jpg
            ...
        painted_lady/
            image001.png
            ...

Example:
    python prepare_data.py --source butterfly_data --output dataset
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"
}
SPLITS = ("train", "val", "test")


def find_images(folder: Path) -> list[Path]:
    """Find supported image files recursively, in a stable order."""
    return sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split two species folders into train/val/test datasets."
    )
    parser.add_argument(
        "--source", type=Path, required=True,
        help="Folder containing exactly two class subfolders, one per species."
    )
    parser.add_argument(
        "--output", type=Path, default=Path("dataset"),
        help="Output dataset folder (default: ./dataset)."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for a repeatable split (default: 42)."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Replace the output folder if it already exists. Use carefully."
    )
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()

    if not source.is_dir():
        print(f"ERROR: source folder does not exist: {source}", file=sys.stderr)
        return 1
    if source == output or source in output.parents:
        print(
            "ERROR: output must be outside the source folder. "
            "For example, use --source butterfly_data --output dataset.",
            file=sys.stderr,
        )
        return 1

    class_dirs = [
        folder for folder in sorted(source.iterdir())
        if folder.is_dir() and find_images(folder)
    ]
    if len(class_dirs) != 2:
        print(
            f"ERROR: expected exactly 2 subfolders containing images in {source}, "
            f"but found {len(class_dirs)}: {[p.name for p in class_dirs]}",
            file=sys.stderr,
        )
        print("The source folder should contain one subfolder per butterfly species.", file=sys.stderr)
        return 1

    images_by_class = {folder.name: find_images(folder) for folder in class_dirs}
    for class_name, files in images_by_class.items():
        if len(files) < 3:
            print(
                f"ERROR: class '{class_name}' has only {len(files)} supported images; "
                "at least 3 are needed.",
                file=sys.stderr,
            )
            return 1

    if output.exists():
        if not args.force:
            print(
                f"ERROR: output folder already exists: {output}\n"
                "Choose another --output path, or add --force to replace it.",
                file=sys.stderr,
            )
            return 1
        if output.is_dir():
            shutil.rmtree(output)
        else:
            output.unlink()

    rng = random.Random(args.seed)
    summary: dict[str, dict[str, int]] = {}

    for class_name, files in images_by_class.items():
        files = files.copy()
        rng.shuffle(files)

        count = len(files)
        n_val = max(1, round(count * 0.15))
        n_test = max(1, round(count * 0.15))
        n_train = count - n_val - n_test
        if n_train < 1:
            print(f"ERROR: not enough images for all splits in class '{class_name}'.", file=sys.stderr)
            return 1

        split_files = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:],
        }

        for split_name in SPLITS:
            destination = output / split_name / class_name
            destination.mkdir(parents=True, exist_ok=True)
            for index, image_path in enumerate(split_files[split_name]):
                # Numbered names avoid collisions if a class folder contains subfolders
                # with files that share the same basename.
                target_name = f"{index:05d}_{image_path.name}"
                shutil.copy2(image_path, destination / target_name)

        summary[class_name] = {split: len(split_files[split]) for split in SPLITS}

    print(f"Dataset created at: {output}")
    print(f"Random seed: {args.seed}")
    print("\nImage counts:")
    print(f"{'Class':<22} {'train':>7} {'val':>7} {'test':>7} {'total':>7}")
    for class_name, counts in summary.items():
        total = sum(counts.values())
        print(
            f"{class_name:<22} {counts['train']:>7} {counts['val']:>7} "
            f"{counts['test']:>7} {total:>7}"
        )
    print("\nFolder structure created: dataset/{train,val,test}/<class_name>/")
    print("Original photos were copied, not moved or modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
