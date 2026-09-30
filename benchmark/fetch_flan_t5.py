"""Fetch the pinned, safetensors-only FLAN-T5-small checkpoint."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from huggingface_hub import snapshot_download

from benchmark.banking77_llm import MODEL_ID, MODEL_REVISION, MODEL_SHA256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(".cache/research-data/flan-t5-small"))
    args = parser.parse_args()
    path = Path(snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=args.output_dir,
        allow_patterns=[
            "config.json",
            "model.safetensors",
            "special_tokens_map.json",
            "spiece.model",
            "tokenizer.json",
            "tokenizer_config.json",
        ],
    ))
    actual_hash = hashlib.sha256((path / "model.safetensors").read_bytes()).hexdigest()
    if actual_hash != MODEL_SHA256:
        raise ValueError(f"FLAN-T5-small weights checksum mismatch: {actual_hash}")
    print(f"{path}: model.safetensors sha256={actual_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
