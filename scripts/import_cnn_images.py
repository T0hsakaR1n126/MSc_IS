#!/usr/bin/env python3
"""Stream the prepared CNN manifest and planar RGB files as MonetDB import SQL."""

import argparse
import csv
from pathlib import Path
import sys


PLANE_SIZE = 64 * 64
IMAGE_SIZE = 3 * PLANE_SIZE
SPLITS = {"train", "validation", "test"}


def read_manifest(root: Path) -> list[tuple[int, str, str, Path]]:
    images = []
    seen_ids = set()
    found_splits = set()
    manifest = root / "manifest.csv"
    with manifest.open(encoding="utf-8", newline="") as file:
        for line_number, row in enumerate(csv.DictReader(file, strict=True), start=2):
            try:
                image_id = int(row["ImageId"])
                if not 0 <= image_id <= 2**63 - 1 or image_id in seen_ids:
                    raise ValueError(f"invalid or duplicate ImageId {image_id}")
                split, label = row["Split"], row["Label"]
                if split not in SPLITS or label not in ("0", "1"):
                    raise ValueError(f"invalid split or label for ImageId {image_id}")
                if (row["Height"], row["Width"], row["Channels"]) != ("64", "64", "3"):
                    raise ValueError(f"ImageId {image_id} must be 64x64 RGB")
                relative_path = f"pixels/{image_id}.rgb"
                if row["File"] != relative_path:
                    raise ValueError(f"unexpected pixel path for ImageId {image_id}")
                path = root / relative_path
                if not path.is_file() or path.stat().st_size != IMAGE_SIZE:
                    raise ValueError(f"{path} must contain exactly {IMAGE_SIZE} bytes")
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{manifest}:{line_number}: {error}") from error
            images.append((image_id, split, label, path))
            seen_ids.add(image_id)
            found_splits.add(split)
    if found_splits != SPLITS:
        raise ValueError(f"{manifest}: missing data for {', '.join(sorted(SPLITS - found_splits))}")
    return images


def emit_import(images: list[tuple[int, str, str, Path]]) -> None:
    count = len(images)
    print(f"""START TRANSACTION;
CREATE LOCAL TEMPORARY TABLE cnn_incoming (
    ImageId BIGINT PRIMARY KEY, Split VARCHAR(16) NOT NULL, Label INTEGER NOT NULL,
    Red BLOB NOT NULL, Green BLOB NOT NULL, Blue BLOB NOT NULL
) ON COMMIT DROP;
COPY {count} RECORDS INTO cnn_incoming FROM STDIN USING DELIMITERS E'\\t', E'\\n', '"';""")
    for image_id, split, label, path in images:
        payload = path.read_bytes()
        if len(payload) != IMAGE_SIZE:
            raise ValueError(f"{path} changed during import")
        planes = [payload[offset:offset + PLANE_SIZE].hex() for offset in range(0, IMAGE_SIZE, PLANE_SIZE)]
        print(image_id, split, label, *planes, sep="\t")

    # Reuse a populated database only if the entire stored corpus matches this manifest.
    print(f"""CREATE LOCAL TEMPORARY TABLE cnn_incoming_refs (
    ImageId BIGINT, Color INTEGER,
    ChannelRef BIGINT NOT NULL DEFAULT NEXT VALUE FOR cnn.image_channel_ref_seq,
    PRIMARY KEY (ImageId, Color)
) ON COMMIT DROP;
INSERT INTO cnn_incoming_refs (ImageId, Color)
SELECT ImageId, Color
FROM cnn_incoming CROSS JOIN (VALUES (0), (1), (2)) AS rgb(Color)
WHERE NOT EXISTS (SELECT 1 FROM cnn.TrainingImage);

INSERT INTO cnn.ImageChannel (ChannelRef, ChannelValue)
SELECT ref.ChannelRef, src.Red
FROM cnn_incoming src JOIN cnn_incoming_refs ref ON src.ImageId = ref.ImageId AND ref.Color = 0
UNION ALL
SELECT ref.ChannelRef, src.Green
FROM cnn_incoming src JOIN cnn_incoming_refs ref ON src.ImageId = ref.ImageId AND ref.Color = 1
UNION ALL
SELECT ref.ChannelRef, src.Blue
FROM cnn_incoming src JOIN cnn_incoming_refs ref ON src.ImageId = ref.ImageId AND ref.Color = 2;

INSERT INTO cnn.TrainingImage (ImageId, Split, Label, Height, Width, Channels, RRef, GRef, BRef)
SELECT src.ImageId, src.Split, src.Label, 64, 64, 3, r.ChannelRef, g.ChannelRef, b.ChannelRef
FROM cnn_incoming src
JOIN cnn_incoming_refs r ON src.ImageId = r.ImageId AND r.Color = 0
JOIN cnn_incoming_refs g ON src.ImageId = g.ImageId AND g.Color = 1
JOIN cnn_incoming_refs b ON src.ImageId = b.ImageId AND b.Color = 2;

CREATE LOCAL TEMPORARY TABLE cnn_import_check (
    ImageCount BIGINT NOT NULL CHECK (ImageCount = {count}),
    MatchingImages BIGINT NOT NULL CHECK (MatchingImages = {count}),
    ChannelCount BIGINT NOT NULL CHECK (ChannelCount = {3 * count})
) ON COMMIT DROP;
INSERT INTO cnn_import_check
SELECT (SELECT COUNT(*) FROM cnn.TrainingImage), COUNT(*), (SELECT COUNT(*) FROM cnn.ImageChannel)
FROM cnn_incoming src JOIN cnn.TrainingImage i ON src.ImageId = i.ImageId
JOIN cnn.ImageChannel r ON i.RRef = r.ChannelRef
JOIN cnn.ImageChannel g ON i.GRef = g.ChannelRef
JOIN cnn.ImageChannel b ON i.BRef = b.ChannelRef
WHERE src.Split = i.Split AND src.Label = i.Label
  AND i.Height = 64 AND i.Width = 64 AND i.Channels = 3
  AND r.ChannelValue = src.Red AND g.ChannelValue = src.Green AND b.ChannelValue = src.Blue;
COMMIT;
SELECT Split, Label, COUNT(*) AS Images FROM cnn.TrainingImage GROUP BY Split, Label ORDER BY Split, Label;""")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="directory with manifest.csv and pixels/")
    args = parser.parse_args()
    try:
        images = read_manifest(args.root)
        print(f"Importing or verifying {len(images)} prepared images.", file=sys.stderr)
        emit_import(images)
        sys.stdout.flush()
    except (OSError, UnicodeError, ValueError, csv.Error) as error:
        parser.exit(1, f"CNN data import: {error}\n")


if __name__ == "__main__":
    main()
