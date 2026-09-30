"""Benchmark frozen MiniLM embeddings with small per-intent binary heads."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score

from benchmark.banking77 import read_dataset
from sempred.fewshot import (
    DEFAULT_ENCODER_ID,
    DEFAULT_ENCODER_REVISION,
    FewShotSemPred,
    FrozenTextEncoder,
)


def fit_heads(
    vectors: np.ndarray,
    labels: list[str],
    categories: list[str],
    *,
    shots: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one balanced binary head per intent, using only train rows."""
    rng = np.random.default_rng(seed)
    coefficients: list[np.ndarray] = []
    intercepts: list[float] = []
    labels_array = np.asarray(labels)
    per_class = shots // 2
    for category in categories:
        positive = np.flatnonzero(labels_array == category)
        negative = np.flatnonzero(labels_array != category)
        selected = np.concatenate(
            (
                rng.choice(positive, per_class, replace=False),
                rng.choice(negative, per_class, replace=False),
            )
        )
        target = (labels_array[selected] == category).astype(np.int8)
        head = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=1000, solver="liblinear"
        )
        head.fit(vectors[selected], target)
        coefficients.append(head.coef_[0].astype(np.float32))
        intercepts.append(float(head.intercept_[0]))
    return np.stack(coefficients), np.asarray(
        intercepts, dtype=np.float32
    )


def ranking_scores(
    vectors: np.ndarray, coefficients: np.ndarray, intercepts: np.ndarray
) -> np.ndarray:
    logits = vectors @ coefficients.T + intercepts[None, :]
    return (1.0 / (1.0 + np.exp(-np.clip(logits, -80, 80)))).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--dataset-card", type=Path, required=True)
    parser.add_argument(
        "--encoder",
        type=Path,
        help="local pinned encoder directory; otherwise use the HF cache",
    )
    parser.add_argument(
        "--vector-cache",
        type=Path,
        default=Path(".cache/research-data/banking77-minilm-vectors.npz"),
    )
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.seeds < 1 or args.torch_threads < 1:
        parser.error("--seeds and --torch-threads must be positive")

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
    encoder_path = args.encoder or DEFAULT_ENCODER_ID
    encoder = FrozenTextEncoder(
        encoder_path,
        revision=None if args.encoder else DEFAULT_ENCODER_REVISION,
        device="cpu",
    )
    if (
        encoder.model_name != DEFAULT_ENCODER_ID
        or encoder.revision != DEFAULT_ENCODER_REVISION
    ):
        raise ValueError("the benchmark requires the pinned all-MiniLM-L6-v2 revision")
    cache_meta = {
        "train_sha256": hashlib.sha256(args.train_data.read_bytes()).hexdigest(),
        "test_sha256": hashlib.sha256(args.test_data.read_bytes()).hexdigest(),
        "encoder": DEFAULT_ENCODER_ID,
        "revision": DEFAULT_ENCODER_REVISION,
        "pooling": "attention-mask mean pooling; L2 normalized",
    }
    cache_path = args.vector_cache
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as stored:
            metadata = json.loads(str(stored["metadata"].item()))
            cached_texts = stored["texts"].tolist()
            vectors = np.asarray(stored["vectors"], dtype=np.float32)
        if (
            metadata != cache_meta
            or cached_texts != unique_texts
            or vectors.shape != (len(unique_texts), encoder.dimension)
        ):
            raise ValueError(
                f"vector cache does not match current dataset/model: {cache_path}"
            )
        embedding_seconds = 0.0
    else:
        started = time.perf_counter()
        vectors = encoder.encode(unique_texts)
        embedding_seconds = time.perf_counter() - started
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            metadata=np.asarray(json.dumps(cache_meta)),
            texts=np.asarray(unique_texts),
            vectors=vectors,
        )

    vector_by_text = {text: vectors[index] for index, text in enumerate(unique_texts)}
    train_vectors = np.stack([vector_by_text[text] for text in train_texts])
    test_vectors = np.stack([vector_by_text[text] for text in test_texts])
    expected = np.asarray([categories.index(label) for label in test_labels])
    results: dict[str, list[dict[str, float]]] = {"8": [], "16": []}
    for shots in (8, 16):
        for seed in range(args.seeds):
            started = time.perf_counter()
            coefficients, intercepts = fit_heads(
                train_vectors, train_labels, categories, shots=shots, seed=seed
            )
            fit_seconds = time.perf_counter() - started
            scoring_started = time.perf_counter()
            scores = ranking_scores(test_vectors, coefficients, intercepts)
            scoring_seconds = time.perf_counter() - scoring_started
            predicted = np.argmax(scores, axis=1)
            results[str(shots)].append(
                {
                    "seed": seed,
                    "accuracy": float(np.mean(predicted == expected)),
                    "macro_f1": float(f1_score(expected, predicted, average="macro")),
                    "balanced_accuracy": float(
                        balanced_accuracy_score(expected, predicted)
                    ),
                    "fit_seconds": fit_seconds,
                    "ticket_rows_per_second_all_intents": len(test_texts)
                    / scoring_seconds,
                    "ticket_predicate_scores_per_second": len(test_texts)
                    * len(categories)
                    / scoring_seconds,
                    "scoring_seconds": scoring_seconds,
                }
            )

    # Construct a deployment-shaped model from one 16-shot seed and measure DuckDB UDF throughput.
    seed = 0
    labels_array = np.asarray(train_labels)
    rng = np.random.default_rng(seed)
    head_params = {}
    for category in categories:
        pos = np.flatnonzero(labels_array == category)
        neg = np.flatnonzero(labels_array != category)
        chosen = np.concatenate(
            (rng.choice(pos, 8, replace=False), rng.choice(neg, 8, replace=False))
        )
        target = (labels_array[chosen] == category).astype(np.int8)
        estimator = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=1000, solver="liblinear"
        )
        estimator.fit(train_vectors[chosen], target)
        head_params[category.replace("_", " ")] = (
            estimator.coef_[0],
            float(estimator.intercept_[0]),
        )
    # Exclude the one-time benchmark corpus embedding pass from the DuckDB
    # request counters: this run measures only work performed by the UDF.
    encoder.texts_encoded = 0
    encoder.encoder_batches = 0
    model = FewShotSemPred(encoder, head_params, abstain_margin=0.1, cache_size=10000)
    test_predicates = [category.replace("_", " ") for category in categories]
    # Arrow UDF operates on candidate rows. Put rows condition-major so bounded cache covers text reuse.
    import duckdb
    import pyarrow as pa

    from sempred.duckdb import register

    con = register(duckdb.connect(), model)
    rows = pa.table(
        {
            "text": pa.array(test_texts * len(test_predicates)),
            "predicate": pa.array(
                [predicate for predicate in test_predicates for _ in test_texts]
            ),
        }
    )
    con.register("benchmark_rows", rows)
    query_started = time.perf_counter()
    sql_counts = con.execute("""
        SELECT count(*), count_if(SEM_SCORE(text, predicate) IS NOT NULL),
               count_if(SEM_PREDICT(text, predicate) IS NULL)
        FROM benchmark_rows
    """).fetchone()
    duckdb_seconds = time.perf_counter() - query_started
    summary = {}
    for shots, runs in results.items():
        for metric in (
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "ticket_rows_per_second_all_intents",
            "ticket_predicate_scores_per_second",
            "fit_seconds",
        ):
            values = np.asarray([run[metric] for run in runs])
            summary.setdefault(shots, {})[metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            }
    report = {
        "dataset": "PolyAI Banking77 official train/test split",
        "dataset_revision": "796a4623935746f71378f0ebd435635a8ce08e50",
        "train_examples": len(train_texts),
        "test_examples": len(test_texts),
        "intents": len(categories),
        "support_sampling": "balanced per predicate from official train split; half positive and half sampled from other intents",
        "seeds": args.seeds,
        "torch_threads": args.torch_threads,
        "encoder": cache_meta,
        "unique_texts_embedded": len(unique_texts),
        "embedding_seconds_first_run": embedding_seconds,
        "embedding_rows_per_second_first_run": len(unique_texts) / embedding_seconds
        if embedding_seconds
        else None,
        "fewshot": {"runs": results, "summary": summary},
        "duckdb": {
            "candidate_pairs": sql_counts[0],
            "scored_pairs": sql_counts[1],
            "unique_ticket_texts": len(set(test_texts)),
            "abstentions": sql_counts[2],
            "seconds": duckdb_seconds,
            "pairs_per_second": sql_counts[0] / duckdb_seconds,
            "unique_ticket_texts_per_second": len(set(test_texts)) / duckdb_seconds,
            "texts_encoded_by_udf": model.encoder.texts_encoded,
            "cache": model.cache_info(),
        },
        "baseline_comparison": {
            "tfidf_full_77_class_accuracy": 0.8545,
            "comparison_rule": "16-shot mean accuracy compared without test-set tuning",
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
