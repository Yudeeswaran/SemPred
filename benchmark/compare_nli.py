"""Explore arithmetic blends of two local NLI checkpoints on one JSONL set.

This is a diagnostic tool. The best blend and threshold are descriptive of the
provided dataset and must not be presented as independent validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from benchmark.metrics import evaluate
from benchmark.model_loading import load_nli_model
from benchmark.run_nli import read_jsonl
from sempred.nli import NLISemPred


def nli_probability(model: NLISemPred, rows: list[dict]) -> np.ndarray:
    output = np.zeros(len(rows), dtype=float)
    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(row["predicate"], []).append(index)
    for predicate, indexes in grouped.items():
        for start in range(0, len(indexes), model.batch_size):
            selected = indexes[start : start + model.batch_size]
            scores = model.predict_batch([rows[index]["text"] for index in selected], predicate)
            output[selected] = [score.probability for score in scores]
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-a", type=Path, required=True, help="first local SemPred NLI checkpoint")
    parser.add_argument("--model-b", type=Path, required=True, help="second local SemPred NLI checkpoint")
    parser.add_argument("--revision-a", help="pinned revision for a raw local HF checkpoint")
    parser.add_argument("--revision-b", help="pinned revision for a raw local HF checkpoint")
    parser.add_argument("--hypothesis-template-a", help="override model A wording with one {predicate} placeholder")
    parser.add_argument("--hypothesis-template-b", help="override model B wording with one {predicate} placeholder")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    rows = read_jsonl(args.dataset)
    labels = [int(row["label"]) for row in rows]
    first = load_nli_model(args.model_a, device=args.device, revision=args.revision_a)
    second = load_nli_model(args.model_b, device=args.device, revision=args.revision_b)
    for model, template, option in (
        (first, args.hypothesis_template_a, "--hypothesis-template-a"),
        (second, args.hypothesis_template_b, "--hypothesis-template-b"),
    ):
        if template is not None:
            if template.count("{predicate}") != 1:
                parser.error(f"{option} must contain exactly one {{predicate}} placeholder")
            model.hypothesis_template = template
    first.batch_size = second.batch_size = args.batch_size
    started = time.perf_counter()
    first_scores = nli_probability(first, rows)
    second_scores = nli_probability(second, rows)

    candidates = {}
    for weight in np.linspace(0.0, 1.0, 11):
        blended = (1 - weight) * first_scores + weight * second_scores
        for threshold in np.linspace(0.05, 0.95, 19):
            metrics = evaluate(labels, blended, threshold=float(threshold))
            candidates[f"{weight:.1f}@{threshold:.2f}"] = {
                "model_b_weight": round(float(weight), 1),
                "threshold": round(float(threshold), 2),
                **{name: metrics[name] for name in ("accuracy", "precision", "recall", "f1", "roc_auc")},
            }
    best = max(candidates.values(), key=lambda result: result["accuracy"])
    report = {
        "dataset": str(args.dataset),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "examples": len(rows),
        "model_a": str(args.model_a),
        "model_a_template": first.hypothesis_template,
        "model_b": str(args.model_b),
        "model_b_template": second.hypothesis_template,
        "weight_definition": "model_b arithmetic probability contribution",
        "best_by_this_dataset_accuracy": best,
        "grid": candidates,
        "warning": "Exploratory blend and threshold selection on this benchmark is not independent validation.",
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "grid"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
