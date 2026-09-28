"""Compare local TF-IDF and NLI probability blends on a JSONL benchmark.

All model artifacts are local trusted files. Use alpha as the NLI contribution:
0 is the classical model average and 1 is the fine-tuned NLI model.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from benchmark.metrics import evaluate
from benchmark.model_loading import load_nli_model
from benchmark.run_nli import read_jsonl
from sempred.nli import NLISemPred


def classical_probability(variant: str, rows: list[dict]) -> np.ndarray:
    artifact = Path("artifacts") / f"{variant}.pkl"
    with artifact.open("rb") as source:
        model, word, char = pickle.load(source)
    pairs = [f"{row['predicate']} [SEP] {row['text']}" for row in rows]
    if variant == "v4_word":
        features = word.transform(pairs)
    elif variant == "v4_char":
        features = char.transform(pairs)
    else:
        from scipy.sparse import hstack

        features = hstack([word.transform(pairs), char.transform(pairs)]).tocsr()
    return model.predict_proba(features)[:, 1]


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
    parser.add_argument("--model", type=Path, required=True, help="local fine-tuned NLI model")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--revision", help="pinned revision for a local base Hugging Face checkpoint")
    parser.add_argument("--classical", choices=("v4_word", "v4_char", "v4_hybrid", "v4_svm"), default="v4_hybrid")
    args = parser.parse_args()

    rows = read_jsonl(args.dataset)
    labels = [int(row["label"]) for row in rows]
    nli = load_nli_model(args.model, device=args.device, revision=args.revision)
    start = time.perf_counter()
    classical = classical_probability(args.classical, rows)
    neural = nli_probability(nli, rows)
    predictions = {}
    for alpha in np.linspace(0.0, 1.0, 11):
        blended = (1.0 - alpha) * classical + alpha * neural
        predictions[f"{alpha:.1f}"] = evaluate(labels, blended, threshold=0.5)
    best_alpha = max(predictions, key=lambda alpha: predictions[alpha]["accuracy"])
    report = {
        "dataset": str(args.dataset),
        "examples": len(rows),
        "classical_model": args.classical,
        "nli_model": nli.model_id,
        "nli_revision": nli.revision,
        "hypothesis_template": nli.hypothesis_template,
        "alpha_definition": "NLI probability contribution in the arithmetic blend",
        "blend_results_threshold_0_5": predictions,
        "best_alpha_by_this_dataset_accuracy": best_alpha,
        "warning": "Exploratory model selection on this benchmark is not independent validation.",
        "elapsed_seconds": time.perf_counter() - start,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
