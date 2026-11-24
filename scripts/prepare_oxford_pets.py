#!/usr/bin/env python3
"""Prepare Oxford-IIIT Pet images as 64×64 planar RGB for the MonetDB CNN."""

import argparse
from collections import Counter, defaultdict
import csv
from pathlib import Path
import random
import re
from tempfile import TemporaryDirectory

from PIL import Image, ImageOps


IMAGE_SIZE = 64
SPECIES_LABEL = {1: 0, 2: 1}  # Oxford: 1=cat, 2=dog; CNN: 0=cat, 1=dog.


def read_annotations(path: Path) -> dict[str, tuple[int, int]]:
    """Return image name -> (breed class ID, binary label)."""
    samples = {}
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                name, class_id, species, breed_id = line.split()
                class_id, species, breed_id = map(int, (class_id, species, breed_id))
                label = SPECIES_LABEL[species]
                if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or class_id <= 0 or breed_id <= 0:
                    raise ValueError("invalid image name or breed ID")
                if name in samples:
                    raise ValueError(f"duplicate image {name}")
            except (ValueError, KeyError) as error:
                raise ValueError(f"{path}:{line_number}: invalid annotation {line!r}") from error
            samples[name] = (class_id, label)
    if not samples:
        raise ValueError(f"{path}: no images listed")
    return samples


def validation_names(trainval: dict[str, tuple[int, int]], fraction: float, seed: int) -> set[str]:
    by_breed = defaultdict(list)
    for name, (class_id, _) in sorted(trainval.items()):
        by_breed[class_id].append(name)

    rng = random.Random(seed)
    selected = set()
    for class_id in sorted(by_breed):
        names = by_breed[class_id]
        if len(names) < 2:
            continue
        rng.shuffle(names)
        count = max(1, min(len(names) - 1, round(len(names) * fraction)))
        selected.update(names[:count])
    return selected


def read_rgb_planes(path: Path) -> bytes:
    try:
        with Image.open(path) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            square = ImageOps.fit(rgb, (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
            return b"".join(channel.tobytes() for channel in square.split())
    except (OSError, Image.DecompressionBombError) as error:
        raise ValueError(f"cannot process image {path}: {error}") from error


def prepare(root: Path, output: Path, fraction: float, seed: int) -> None:
    if output.exists() or output.is_symlink():
        raise ValueError(f"output already exists: {output}; choose a new directory")
    images = root / "images"
    if not images.is_dir():
        raise ValueError(f"image directory missing: {images}")
    trainval = read_annotations(root / "annotations/trainval.txt")
    test = read_annotations(root / "annotations/test.txt")
    overlap = trainval.keys() & test.keys()
    if overlap:
        raise ValueError(f"image listed in both official splits: {min(overlap)}")
    validation = validation_names(trainval, fraction, seed)
    counts = Counter()

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".pets-", dir=output.parent) as temp:
        staged = Path(temp) / "dataset"
        (staged / "pixels").mkdir(parents=True)
        with (staged / "manifest.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(("ImageId", "Split", "Label", "Height", "Width", "Channels", "File", "SourceImage"))
            for image_id, (name, (_, label)) in enumerate(sorted((trainval | test).items()), start=1):
                if name in test:
                    split = "test"
                elif name in validation:
                    split = "validation"
                else:
                    split = "train"
                source = f"{name}.jpg"
                relative_path = f"pixels/{image_id}.rgb"
                (staged / relative_path).write_bytes(read_rgb_planes(images / source))
                writer.writerow((image_id, split, label, IMAGE_SIZE, IMAGE_SIZE, 3, relative_path, source))
                counts[split, label] += 1
        staged.rename(output)

    print(f"Manifest: {output / 'manifest.csv'}")
    for split in ("train", "validation", "test"):
        print(f"{split:10s} cats={counts[split, 0]:4d}  dogs={counts[split, 1]:4d}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="directory with images/ and annotations/")
    parser.add_argument("--output", required=True, type=Path, help="new directory for manifest.csv and pixels/")
    parser.add_argument("--validation-fraction", type=float, default=0.2, help="fraction of official trainval per breed (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="validation split seed (default: 42)")
    args = parser.parse_args()
    if not 0.0 < args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be between 0 and 1")
    try:
        prepare(args.root, args.output, args.validation_fraction, args.seed)
    except (OSError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
