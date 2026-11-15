"""Write or verify per-image checksums for a generated dataset.

Each image is hashed as SHA-256 over its decoded pixels (mode, size, raw
bytes), not over the PNG file: different Pillow/zlib builds can encode
identical pixels into different PNG bytes, which would make a file-level
hash fail on a correctly regenerated dataset.

    python checksums.py write              # dataset/ -> reference/
    python checksums.py verify             # check dataset/ against reference/
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

CHECKSUM_FILE = "image_sha256.csv"


def pixel_sha256(path: Path) -> str:
    with Image.open(path) as img:
        h = hashlib.sha256(f"{img.mode}:{img.width}x{img.height}:".encode())
        h.update(img.tobytes())
    return h.hexdigest()


def image_paths(dataset_dir: Path) -> list[str]:
    meta = pd.read_csv(dataset_dir / "metadata.csv")
    return sorted(meta["clean_image"].tolist() + meta["error_image"].tolist())


def write(dataset_dir: Path, reference_dir: Path) -> None:
    reference_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"image": p, "sha256": pixel_sha256(dataset_dir / p)} for p in image_paths(dataset_dir)]
    pd.DataFrame(rows).to_csv(reference_dir / CHECKSUM_FILE, index=False)
    shutil.copyfile(dataset_dir / "metadata.csv", reference_dir / "metadata.csv")
    print(f"Wrote {len(rows)} checksums and metadata.csv to {reference_dir}/")


def verify(dataset_dir: Path, reference_dir: Path) -> bool:
    ok = True
    ref_meta = pd.read_csv(reference_dir / "metadata.csv")
    meta = pd.read_csv(dataset_dir / "metadata.csv")
    if not meta.equals(ref_meta):
        print("metadata.csv differs from the reference")
        ok = False

    expected = dict(pd.read_csv(reference_dir / CHECKSUM_FILE).itertuples(index=False))
    n_bad = 0
    for p, digest in expected.items():
        path = dataset_dir / p
        if not path.exists():
            print(f"missing: {p}")
            n_bad += 1
        elif pixel_sha256(path) != digest:
            print(f"mismatch: {p}")
            n_bad += 1
    extra = set(image_paths(dataset_dir)) - expected.keys()
    for p in sorted(extra):
        print(f"unexpected: {p}")
    ok = ok and n_bad == 0 and not extra

    print(
        f"{len(expected) - n_bad}/{len(expected)} images match"
        + ("; dataset matches the reference." if ok else "; dataset does NOT match the reference.")
    )
    return ok


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["write", "verify"])
    p.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    p.add_argument("--reference-dir", type=Path, default=Path("reference"))
    a = p.parse_args()
    if a.command == "write":
        write(a.dataset_dir, a.reference_dir)
    else:
        sys.exit(0 if verify(a.dataset_dir, a.reference_dir) else 1)


if __name__ == "__main__":
    main()
