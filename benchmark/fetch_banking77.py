"""Fetch the pinned Banking77 train/test splits and dataset card into a local cache."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

from benchmark.banking77 import (
    DATASET_CARD_SHA256,
    DATASET_REVISION,
    DATASET_SHA256,
    TRAIN_SHA256,
)

BASE_URL = f"https://huggingface.co/datasets/PolyAI/banking77/resolve/{DATASET_REVISION}"
FILES = {
    "banking77_train.parquet": (f"{BASE_URL}/data/train-00000-of-00001.parquet", TRAIN_SHA256),
    "banking77_test.parquet": (f"{BASE_URL}/data/test-00000-of-00001.parquet", DATASET_SHA256),
    "banking77_README.md": (f"{BASE_URL}/README.md", DATASET_CARD_SHA256),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(".cache/research-data"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, (url, expected_hash) in FILES.items():
        destination = args.output_dir / filename
        with urlopen(url, timeout=60) as response:
            content = response.read()
        actual_hash = hashlib.sha256(content).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"{filename} checksum mismatch: {actual_hash}")
        destination.write_bytes(content)
        print(f"{destination}: sha256={actual_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
