"""Command-line workflow for training and using a local SemPred model."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from . import SemPred
from .nli import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION, NLISemPred


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number}: each JSONL row must be an object")
            rows.append(row)
    return rows


def _train(args: argparse.Namespace) -> int:
    rows = _read_jsonl(args.data)
    required = {"text", "predicate", "label"}
    for i, row in enumerate(rows, 1):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"{args.data}: row {i} is missing {', '.join(sorted(missing))}")
    if args.backend == "nli":
        if not args.base_model.is_dir():
            raise ValueError(
                f"NLI checkpoint not found at {args.base_model}; first run "
                f"`sempred download-nli --output {args.base_model}`"
            )
        command = [
            sys.executable,
            "-m",
            "training.finetune_nli",
            "--base-model",
            str(args.base_model),
            "--train-data",
            str(args.data),
            "--output",
            str(args.model),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--learning-rate",
            str(args.learning_rate),
            "--max-length",
            str(args.max_length),
            "--validation-split",
            args.validation_split,
            "--validation-group-count",
            str(args.validation_group_count),
            "--negative-nli-label",
            args.negative_nli_label,
            "--device",
            args.device,
        ]
        if args.seed is not None:
            command.extend(("--seed", str(args.seed)))
        return subprocess.run(command, check=False).returncode

    model = SemPred.fit(
        [row["text"] for row in rows],
        [row["predicate"] for row in rows],
        [row["label"] for row in rows],
        threshold=args.threshold,
        abstain_margin=args.abstain_margin,
    )
    model.save(args.model)
    print(json.dumps({"model": str(args.model), "examples": len(rows), "backend": "tfidf-baseline"}))
    return 0


def _predict(args: argparse.Namespace) -> int:
    if args.model.is_dir():
        model = NLISemPred.load(args.model, device=args.device)
    else:
        model = SemPred.load(args.model)
    result = model.predict(args.text, args.predicate)
    print(json.dumps(asdict(result)))
    return 0


def _download_nli(args: argparse.Namespace) -> int:
    model = NLISemPred.from_pretrained(
        args.model_id,
        revision=args.revision,
        device=args.device,
        cache_dir=args.cache_dir,
        threshold=args.threshold,
        abstain_margin=args.abstain_margin,
        max_length=args.max_length,
        batch_size=args.batch_size,
        hypothesis_template=args.hypothesis_template,
    )
    model.save(args.output)
    print(json.dumps({
        "model_directory": str(args.output),
        "model_id": args.model_id,
        "revision": args.revision,
        "backend": "nli-cross-encoder",
    }))
    return 0


def _calibrate(args: argparse.Namespace) -> int:
    rows = _read_jsonl(args.data)
    required = {"text", "predicate", "label"}
    for i, row in enumerate(rows, 1):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"{args.data}: row {i} is missing {', '.join(sorted(missing))}")
    if args.family:
        families = set(args.family)
        rows = [row for row in rows if row.get("family") in families]
        if not rows:
            raise ValueError("no calibration rows matched --family")
    model = NLISemPred.load(args.model, device=args.device)
    scale, bias = model.calibrate(
        [row["text"] for row in rows],
        [row["predicate"] for row in rows],
        [row["label"] for row in rows],
    )
    model.calibration_metadata = {
        "source_dataset": str(args.data),
        "source_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "examples_used": len(rows),
        "families_used": sorted({row["family"] for row in rows if "family" in row}),
    }
    model.save(args.output)
    print(json.dumps({"calibration_examples": len(rows), "scale": scale, "bias": bias, "saved_to": str(args.output)}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sempred", description="Train and query a local semantic-predicate model.")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="fine-tune an NLI encoder from labeled JSONL")
    train.add_argument("--data", type=Path, required=True, help="JSONL with text, predicate, and binary label fields")
    train.add_argument("--model", type=Path, required=True, help="output path for the trained model")
    train.add_argument("--backend", choices=("nli", "tfidf"), default="nli", help="use the semantic encoder by default; tfidf is a baseline")
    train.add_argument("--base-model", type=Path, default=Path("models/sempred-nli"), help="local checkpoint directory saved by download-nli")
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--learning-rate", type=float, default=2e-5)
    train.add_argument("--max-length", type=int, default=128)
    train.add_argument("--negative-nli-label", choices=("neutral", "contradiction"), default="neutral")
    train.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--validation-split", choices=("auto", "construction", "family"), default="auto")
    train.add_argument("--validation-group-count", type=int, default=2)
    train.add_argument("--threshold", type=float, default=0.5, help="decision threshold for the TF-IDF baseline")
    train.add_argument("--abstain-margin", type=float, default=0.0, help="abstention margin for the TF-IDF baseline")
    train.set_defaults(handler=_train)

    predict = commands.add_parser("predict", help="evaluate one text and predicate")
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    predict.add_argument("--text", required=True)
    predict.add_argument("--predicate", required=True)
    predict.set_defaults(handler=_predict)

    download = commands.add_parser("download-nli", help="download and save the pinned pretrained NLI model")
    download.add_argument("--output", type=Path, required=True, help="directory to save model and tokenizer")
    download.add_argument("--cache-dir", type=Path, default=Path(".cache/huggingface"))
    download.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    download.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    download.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    download.add_argument("--threshold", type=float, default=0.5)
    download.add_argument("--abstain-margin", type=float, default=0.0)
    download.add_argument("--max-length", type=int, default=256)
    download.add_argument("--batch-size", type=int, default=32)
    download.add_argument("--hypothesis-template", default="It is true that {predicate}.")
    download.set_defaults(handler=_download_nli)

    calibrate = commands.add_parser("calibrate", help="fit score calibration on a held-out labeled JSONL set")
    calibrate.add_argument("--model", type=Path, required=True, help="saved local NLI model directory")
    calibrate.add_argument("--output", type=Path, required=True, help="directory for the calibrated model copy")
    calibrate.add_argument("--data", type=Path, required=True, help="held-out JSONL with text, predicate, and label")
    calibrate.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    calibrate.add_argument("--family", action="append", default=[], help="optional family filter for the calibration JSONL")
    calibrate.set_defaults(handler=_calibrate)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.handler(args)
    except (OSError, ValueError, KeyError, RuntimeError, TypeError) as exc:
        print(f"sempred: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
