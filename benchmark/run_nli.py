"""Reproducible scoring of an NLI model on a SemPred JSONL benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from benchmark.metrics import evaluate
from benchmark.model_loading import load_nli_model


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    for number, row in enumerate(rows, 1):
        if not all(field in row for field in ("text", "predicate", "label")):
            raise ValueError(f"{path}: row {number} needs text, predicate, and label fields")
    if len({int(row["label"]) for row in rows}) != 2:
        raise ValueError("benchmark data must contain both binary labels")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="saved local SemPred NLI model directory")
    parser.add_argument("--dataset", type=Path, required=True, help="locked JSONL benchmark file")
    parser.add_argument("--output", type=Path, required=True, help="metrics JSON output path")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--revision", help="pinned revision for a local base Hugging Face checkpoint")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, help="override the model decision threshold")
    parser.add_argument("--hypothesis-template", help="optional development override with one {predicate} placeholder")
    parser.add_argument("--holdout-family", action="append", default=[], help="score only matching family values from the source JSONL")
    args = parser.parse_args()

    rows = read_jsonl(args.dataset)
    if args.holdout_family:
        selected_families = set(args.holdout_family)
        rows = [row for row in rows if row.get("family") in selected_families]
        if not rows:
            parser.error("no rows matched --holdout-family")
    model_load_started = time.perf_counter()
    model = load_nli_model(args.model, device=args.device, revision=args.revision)
    model_load_seconds = time.perf_counter() - model_load_started
    model.batch_size = args.batch_size
    if args.threshold is not None:
        if not 0.0 <= args.threshold <= 1.0:
            parser.error("--threshold must be between 0 and 1")
        model.threshold = args.threshold
    if args.hypothesis_template is not None:
        if args.hypothesis_template.count("{predicate}") != 1:
            parser.error("--hypothesis-template must contain exactly one {predicate} placeholder")
        model.hypothesis_template = args.hypothesis_template
    texts = [row["text"] for row in rows]
    predicates = [row["predicate"] for row in rows]
    labels = [int(row["label"]) for row in rows]

    started = time.perf_counter()
    probabilities = [0.0] * len(rows)
    grouped: dict[str, list[int]] = {}
    for index, predicate in enumerate(predicates):
        grouped.setdefault(predicate, []).append(index)
    for predicate, indices in grouped.items():
        for start in range(0, len(indices), args.batch_size):
            selected = indices[start : start + args.batch_size]
            batch = model.predict_batch([texts[index] for index in selected], predicate)
            for index, item in zip(selected, batch):
                probabilities[index] = item.probability
    elapsed = time.perf_counter() - started
    predictions = [value >= model.threshold for value in probabilities]
    selective = {}
    for margin in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40):
        selected = [abs(value - model.threshold) >= margin for value in probabilities]
        retained = [target for target, keep in zip(labels, selected) if keep]
        predicted = [result for result, keep in zip(predictions, selected) if keep]
        selective[str(margin)] = {
            "coverage": sum(selected) / len(selected),
            "accuracy_on_covered": sum(actual == guess for actual, guess in zip(retained, predicted)) / len(retained) if retained else None,
            "errors_on_covered": sum(actual != guess for actual, guess in zip(retained, predicted)),
        }

    def package_version(name: str) -> str | None:
        try:
            return version(name)
        except PackageNotFoundError:
            return None

    threshold_sweep = {}
    for threshold in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
        metrics = evaluate(labels, probabilities, threshold=threshold)
        threshold_sweep[f"{threshold:.2f}"] = {
            name: metrics[name] for name in ("accuracy", "precision", "recall", "f1")
        }

    result = {
        "model": model.model_id,
        "revision": model.revision,
        "hypothesis_template": model.hypothesis_template,
        "dataset": str(args.dataset),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "families_evaluated": sorted({row["family"] for row in rows if "family" in row}),
        "examples": len(rows),
        "device": args.device,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": package_version("torch"),
        "transformers": package_version("transformers"),
        "metrics": evaluate(labels, probabilities, threshold=model.threshold),
        "threshold_sweep": threshold_sweep,
        "selective": selective,
        "elapsed_seconds": elapsed,
        "model_load_seconds": model_load_seconds,
        "cold_total_seconds": model_load_seconds + elapsed,
        "rows_per_second": len(rows) / elapsed if elapsed else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
