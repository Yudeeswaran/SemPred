"""Run the predeclared k=16 base-encoder Banking77 follow-up."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from benchmark.banking77 import read_dataset
from benchmark.banking77_fewshot import fit_heads, sample_supports, tfidf_scores
from sempred.fewshot import FrozenTextEncoder

ENCODER_ID = "sentence-transformers/all-mpnet-base-v2"
ENCODER_REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"
GAP_CUTOFFS = (0.05, 0.10, 0.15, 0.20, 0.25)


def gap_metrics(scores: np.ndarray, expected: np.ndarray) -> dict[str, dict]:
    ranked = np.sort(scores, axis=1)
    gaps = ranked[:, -1] - ranked[:, -2]
    top = np.argmax(scores, axis=1)
    results = {}
    for cutoff in GAP_CUTOFFS:
        accepted = gaps >= cutoff
        results[f"{cutoff:.2f}"] = {
            "cutoff": cutoff,
            "ticket_coverage": float(np.mean(accepted)),
            "ticket_accuracy": float(np.mean(top[accepted] == expected[accepted]))
            if np.any(accepted)
            else None,
            "committed_tickets": int(accepted.sum()),
        }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--dataset-card", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--vector-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--torch-threads", type=int, default=8)
    args = parser.parse_args()
    if args.seeds != 10:
        parser.error("this predeclared round uses exactly the existing ten seeds")
    if args.torch_threads < 1:
        parser.error("--torch-threads must be positive")

    train_texts, train_labels, categories = read_dataset(
        args.train_data, args.dataset_card, split="train"
    )
    test_texts, test_labels, test_categories = read_dataset(
        args.test_data, args.dataset_card
    )
    if categories != test_categories:
        raise ValueError("train and test intent sets do not match")
    unique_texts = list(dict.fromkeys(train_texts + test_texts))

    import torch

    torch.set_num_threads(args.torch_threads)
    encoder = FrozenTextEncoder(args.encoder, revision=None, device="cpu")
    if (encoder.model_name, encoder.revision) != (ENCODER_ID, ENCODER_REVISION):
        raise ValueError("local encoder metadata does not match the pinned base model")
    cache_meta = {
        "model": ENCODER_ID,
        "revision": ENCODER_REVISION,
        "train_sha256": hashlib.sha256(args.train_data.read_bytes()).hexdigest(),
        "test_sha256": hashlib.sha256(args.test_data.read_bytes()).hexdigest(),
        "pooling": "attention-mask mean pooling; L2 normalized",
        "max_length": encoder.max_length,
    }
    if args.vector_cache.is_file():
        with np.load(args.vector_cache, allow_pickle=False) as cache:
            if json.loads(str(cache["metadata"].item())) != cache_meta:
                raise ValueError("base-model vector cache metadata mismatch")
            if cache["texts"].tolist() != unique_texts:
                raise ValueError("base-model vector cache corpus mismatch")
            vectors = np.asarray(cache["vectors"], dtype=np.float32)
        embedding_seconds = 0.0
    else:
        started = time.perf_counter()
        vectors = encoder.encode(unique_texts)
        embedding_seconds = time.perf_counter() - started
        args.vector_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.vector_cache,
            metadata=np.asarray(json.dumps(cache_meta)),
            texts=np.asarray(unique_texts),
            vectors=vectors,
        )

    vector_by_text = {text: vectors[i] for i, text in enumerate(unique_texts)}
    train_vectors = np.stack([vector_by_text[text] for text in train_texts])
    test_vectors = np.stack([vector_by_text[text] for text in test_texts])
    expected = np.asarray([categories.index(label) for label in test_labels])
    runs = []
    for seed in range(10):
        supports = sample_supports(train_labels, categories, shots=16, seed=seed)
        support_hash = hashlib.sha256(
            json.dumps(
                {key: value.tolist() for key, value in supports.items()},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        started = time.perf_counter()
        coefficients, intercepts = fit_heads(
            train_vectors, train_labels, categories, supports
        )
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        logits = test_vectors @ coefficients.T + intercepts[None, :]
        scores = (1.0 / (1.0 + np.exp(-np.clip(logits, -80, 80)))).astype(
            np.float32
        )
        scoring_seconds = time.perf_counter() - started
        tfidf, tfidf_fit_seconds, tfidf_scoring_seconds = tfidf_scores(
            train_texts, test_texts, train_labels, categories, supports
        )
        predicted = np.argmax(scores, axis=1)
        tfidf_predicted = np.argmax(tfidf, axis=1)
        runs.append(
            {
                "seed": seed,
                "support_sha256": support_hash,
                "top1_accuracy": float(np.mean(predicted == expected)),
                "macro_f1": float(f1_score(expected, predicted, average="macro")),
                "tfidf_same_support_accuracy": float(
                    np.mean(tfidf_predicted == expected)
                ),
                "accuracy_delta_vs_tfidf": float(
                    np.mean(predicted == expected) - np.mean(tfidf_predicted == expected)
                ),
                "fit_seconds": fit_seconds,
                "scoring_seconds": scoring_seconds,
                "tfidf_fit_seconds": tfidf_fit_seconds,
                "tfidf_scoring_seconds": tfidf_scoring_seconds,
                "gap_metrics": gap_metrics(scores, expected),
            }
        )

    mean_accuracy = float(np.mean([row["top1_accuracy"] for row in runs]))
    delta = float(np.mean([row["accuracy_delta_vs_tfidf"] for row in runs]))
    gaps = {
        f"{cutoff:.2f}": {
            "ticket_coverage": float(
                np.mean(
                    [
                        row["gap_metrics"][f"{cutoff:.2f}"]["ticket_coverage"]
                        for row in runs
                    ]
                )
            ),
            "ticket_accuracy": (
                float(
                    np.mean(
                        [
                            row["gap_metrics"][f"{cutoff:.2f}"]["ticket_accuracy"]
                            for row in runs
                            if row["gap_metrics"][f"{cutoff:.2f}"]["ticket_accuracy"]
                            is not None
                        ]
                    )
                )
                if any(
                    row["gap_metrics"][f"{cutoff:.2f}"]["ticket_accuracy"]
                    is not None
                    for row in runs
                )
                else None
            ),
        }
        for cutoff in GAP_CUTOFFS
    }
    gate = any(
        row["ticket_coverage"] >= 0.50 and (row["ticket_accuracy"] or 0) >= 0.90
        for row in gaps.values()
    )
    accuracy_gate = mean_accuracy >= 0.80
    report = {
        "experiment": "base-size frozen encoder, 16 examples per predicate, Banking77",
        "dataset": "PolyAI Banking77 official train/test split",
        "dataset_revision": "796a4623935746f71378f0ebd435635a8ce08e50",
        "train_examples": len(train_texts),
        "test_examples": len(test_texts),
        "intents": len(categories),
        "shots_per_predicate": 16,
        "seeds": 10,
        "support_sampling": "identical implementation, categories, and seeds as the MiniLM and TF-IDF experiment",
        "encoder": cache_meta,
        "encoder_batch_size": encoder.batch_size,
        "torch_threads": args.torch_threads,
        "unique_texts_embedded": len(unique_texts),
        "embedding_seconds_first_run": embedding_seconds,
        "embedding_texts_per_second_first_run": len(unique_texts) / embedding_seconds
        if embedding_seconds
        else None,
        "gap_rule": "rank all 77 predicate sigmoid scores; commit the top predicate only when top1 minus top2 is at least the fixed cutoff",
        "gap_cutoffs": list(GAP_CUTOFFS),
        "runs": runs,
        "summary": {
            "top1_accuracy_mean": mean_accuracy,
            "top1_accuracy_sd": float(np.std([r["top1_accuracy"] for r in runs], ddof=1)),
            "macro_f1_mean": float(np.mean([r["macro_f1"] for r in runs])),
            "macro_f1_sd": float(np.std([r["macro_f1"] for r in runs], ddof=1)),
            "paired_accuracy_delta_vs_same_support_tfidf_mean": delta,
            "paired_accuracy_delta_vs_same_support_tfidf_sd": float(
                np.std([r["accuracy_delta_vs_tfidf"] for r in runs], ddof=1)
            ),
            "top1_top2_gap_metrics": gaps,
        },
        "predeclared_continue_rule": {
            "top1_accuracy": "mean over the ten seeds >= 80%",
            "abstention": "at one fixed gap cutoff from 0.05, 0.10, 0.15, 0.20, 0.25, mean ticket coverage >= 50% and mean accuracy on committed tickets >= 90%",
            "accuracy_gate_passed": accuracy_gate,
            "coverage_accuracy_gate_passed": gate,
            "continue": accuracy_gate and gate,
            "decision": "continue evaluation" if accuracy_gate and gate else "stop this approach and publish the negative result",
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["predeclared_continue_rule"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
