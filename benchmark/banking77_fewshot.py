"""Benchmark frozen MiniLM embeddings with small per-intent binary heads."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer
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
    supports: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one balanced binary head per intent on the supplied support rows."""
    coefficients: list[np.ndarray] = []
    intercepts: list[float] = []
    labels_array = np.asarray(labels)
    for category in categories:
        selected = supports[category]
        target = (labels_array[selected] == category).astype(np.int8)
        head = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=1000, solver="liblinear"
        )
        head.fit(vectors[selected], target)
        coefficients.append(head.coef_[0].astype(np.float32))
        intercepts.append(float(head.intercept_[0]))
    return np.stack(coefficients), np.asarray(intercepts, dtype=np.float32)


def sample_supports(
    labels: list[str], categories: list[str], *, shots: int, seed: int
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels_array = np.asarray(labels)
    supports = {}
    for category in categories:
        positive = np.flatnonzero(labels_array == category)
        negative = np.flatnonzero(labels_array != category)
        supports[category] = np.concatenate(
            (
                rng.choice(positive, shots // 2, replace=False),
                rng.choice(negative, shots // 2, replace=False),
            )
        )
    return supports


def tfidf_scores(
    train_texts: list[str],
    test_texts: list[str],
    labels: list[str],
    categories: list[str],
    supports: dict[str, np.ndarray],
) -> tuple[np.ndarray, float, float]:
    """Fit shared low-shot TF-IDF features and one balanced head per intent."""
    training_indices = np.asarray(
        sorted({int(i) for rows in supports.values() for i in rows})
    )
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=200_000)
    fit_started = time.perf_counter()
    train_vectors = vectorizer.fit_transform(
        [train_texts[index] for index in training_indices]
    )
    local_index = {source: index for index, source in enumerate(training_indices)}
    labels_array = np.asarray(labels)
    coefficients, intercepts = [], []
    for category in categories:
        source_indices = supports[category]
        local_rows = [local_index[int(index)] for index in source_indices]
        target = (labels_array[source_indices] == category).astype(np.int8)
        head = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=1000, solver="liblinear"
        )
        head.fit(train_vectors[local_rows], target)
        coefficients.append(head.coef_[0])
        intercepts.append(float(head.intercept_[0]))
    fit_seconds = time.perf_counter() - fit_started
    score_started = time.perf_counter()
    test_vectors = vectorizer.transform(test_texts)
    logits = (
        np.asarray(test_vectors @ np.stack(coefficients).T)
        + np.asarray(intercepts)[None, :]
    )
    scores = expit(logits).astype(np.float32)
    scoring_seconds = time.perf_counter() - score_started
    return scores, fit_seconds, scoring_seconds


def evaluate_binary_scores(
    scores: np.ndarray,
    labels: list[str],
    categories: list[str],
    margins: tuple[float, ...],
) -> dict[str, dict]:
    truth = np.asarray(labels)[:, None] == np.asarray(categories)[None, :]
    report = {}
    for margin in margins:
        decisions = np.full(scores.shape, -1, dtype=np.int8)
        decisions[scores >= 0.5 + margin] = 1
        decisions[scores < 0.5 - margin] = 0
        accepted = decisions >= 0
        positives = decisions == 1
        true_positive = int(np.sum(positives & truth))
        false_positive = int(np.sum(positives & ~truth))
        false_negative = int(np.sum(truth & ~positives))
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        committed_accuracy = (
            float(np.mean(decisions[accepted] == truth[accepted]))
            if np.any(accepted)
            else None
        )
        per_predicate = {}
        for column, category in enumerate(categories):
            pred = positives[:, column]
            actual = truth[:, column]
            tp = int(np.sum(pred & actual))
            fp = int(np.sum(pred & ~actual))
            fn = int(np.sum(actual & ~pred))
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / (tp + fn) if tp + fn else 0.0
            per_predicate[category.replace("_", " ")] = {
                "precision": p,
                "recall": r,
                "f1": 2 * p * r / (p + r) if p + r else 0.0,
                "coverage": float(np.mean(accepted[:, column])),
            }

        all_known = accepted.all(axis=1)
        true_count = positives.sum(axis=1)
        resolved = all_known & (true_count == 1)
        true_columns = np.argmax(positives, axis=1)
        gold_columns = np.asarray([categories.index(label) for label in labels])
        exact_correct = resolved & (true_columns == gold_columns)
        ranked_columns = np.argmax(scores, axis=1)
        ranked_confident = (
            scores[np.arange(len(scores)), ranked_columns] >= 0.5 + margin
        )
        ranked_correct = ranked_confident & (ranked_columns == gold_columns)
        one_positive = true_count == 1
        one_positive_correct = one_positive & (true_columns == gold_columns)
        report[f"{margin:.2f}"] = {
            "pair_coverage": float(np.mean(accepted)),
            "committed_pair_accuracy": committed_accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "committed_pairs": int(accepted.sum()),
            "pair_count": int(accepted.size),
            "resolved_ticket_coverage": float(np.mean(resolved)),
            "resolved_ticket_accuracy": float(np.sum(exact_correct) / np.sum(resolved))
            if np.any(resolved)
            else None,
            "resolved_tickets": int(resolved.sum()),
            "top_ranked_ticket_coverage": float(np.mean(ranked_confident)),
            "top_ranked_ticket_accuracy": float(
                np.sum(ranked_correct) / np.sum(ranked_confident)
            )
            if np.any(ranked_confident)
            else None,
            "top_ranked_tickets": int(ranked_confident.sum()),
            "single_positive_ticket_coverage": float(np.mean(one_positive)),
            "single_positive_ticket_accuracy": float(
                np.sum(one_positive_correct) / np.sum(one_positive)
            )
            if np.any(one_positive)
            else None,
            "per_predicate": per_predicate,
        }
    return report


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
        "max_length": encoder.max_length,
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
    margins = (0.05, 0.10, 0.15, 0.20, 0.25)
    results: dict[str, dict[str, list[dict]]] = {
        method: {"8": [], "16": []} for method in ("fewshot", "tfidf")
    }
    binary_runs: dict[str, dict[str, list[dict]]] = {
        method: {"8": [], "16": []} for method in ("fewshot", "tfidf")
    }
    for shots in (8, 16):
        for seed in range(args.seeds):
            supports = sample_supports(train_labels, categories, shots=shots, seed=seed)
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
            scoring_started = time.perf_counter()
            embedding_scores = ranking_scores(test_vectors, coefficients, intercepts)
            embedding_scoring_seconds = time.perf_counter() - scoring_started
            tfidf_scores_matrix, tfidf_fit_seconds, tfidf_scoring_seconds = (
                tfidf_scores(
                    train_texts, test_texts, train_labels, categories, supports
                )
            )
            for method, scores, fit_time, scoring_time in (
                ("fewshot", embedding_scores, fit_seconds, embedding_scoring_seconds),
                (
                    "tfidf",
                    tfidf_scores_matrix,
                    tfidf_fit_seconds,
                    tfidf_scoring_seconds,
                ),
            ):
                predicted = np.argmax(scores, axis=1)
                metrics = evaluate_binary_scores(
                    scores, test_labels, categories, margins
                )
                binary_runs[method][str(shots)].append(metrics)
                results[method][str(shots)].append(
                    {
                        "seed": seed,
                        "support_sha256": support_hash,
                        "accuracy": float(np.mean(predicted == expected)),
                        "macro_f1": float(
                            f1_score(expected, predicted, average="macro")
                        ),
                        "balanced_accuracy": float(
                            balanced_accuracy_score(expected, predicted)
                        ),
                        "fit_seconds": fit_time,
                        "ticket_rows_per_second_all_intents": len(test_texts)
                        / scoring_time,
                        "ticket_predicate_scores_per_second": len(test_texts)
                        * len(categories)
                        / scoring_time,
                        "scoring_seconds": scoring_time,
                        "binary_metrics_by_margin": {
                            margin: {
                                key: value
                                for key, value in report.items()
                                if key != "per_predicate"
                            }
                            for margin, report in metrics.items()
                        },
                    }
                )

    performance_summary: dict[str, dict[str, dict]] = {method: {} for method in results}
    binary_summary: dict[str, dict[str, dict]] = {method: {} for method in results}
    for method, shot_results in results.items():
        for shots, runs in shot_results.items():
            performance_summary[method][shots] = {}
            for metric in (
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "ticket_rows_per_second_all_intents",
                "ticket_predicate_scores_per_second",
                "fit_seconds",
            ):
                values = np.asarray([run[metric] for run in runs])
                performance_summary[method][shots][metric] = {
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                }

            binary_summary[method][shots] = {}
            for margin in margins:
                margin_key = f"{margin:.2f}"
                seed_metrics = [run[margin_key] for run in binary_runs[method][shots]]
                seed_intents = [
                    report[margin_key]["per_predicate"]
                    for report in binary_runs[method][shots]
                ]
                aggregate = {}
                for metric in (
                    "pair_coverage",
                    "committed_pair_accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "resolved_ticket_coverage",
                    "resolved_ticket_accuracy",
                    "top_ranked_ticket_coverage",
                    "top_ranked_ticket_accuracy",
                    "single_positive_ticket_coverage",
                    "single_positive_ticket_accuracy",
                ):
                    values = np.asarray(
                        [
                            item[metric]
                            for item in seed_metrics
                            if item[metric] is not None
                        ]
                    )
                    aggregate[metric] = {
                        "mean": float(values.mean()) if len(values) else None,
                        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    }
                per_predicate = {}
                for predicate in categories:
                    name = predicate.replace("_", " ")
                    per_predicate[name] = {}
                    for metric in ("precision", "recall", "f1", "coverage"):
                        values = np.asarray([run[name][metric] for run in seed_intents])
                        per_predicate[name][metric] = {
                            "mean": float(values.mean()),
                            "std": float(values.std(ddof=1))
                            if len(values) > 1
                            else 0.0,
                        }
                aggregate["per_predicate"] = per_predicate
                aggregate["card swallowed"] = per_predicate["card swallowed"]
                binary_summary[method][shots][margin_key] = aggregate

    paired_deltas = {}
    for shots in ("8", "16"):
        embeddings_by_seed = {
            row["seed"]: row["accuracy"] for row in results["fewshot"][shots]
        }
        tfidf_by_seed = {
            row["seed"]: row["accuracy"] for row in results["tfidf"][shots]
        }
        differences = np.asarray(
            [
                embeddings_by_seed[seed] - tfidf_by_seed[seed]
                for seed in embeddings_by_seed
            ]
        )
        paired_deltas[shots] = {
            "embedding_minus_tfidf_accuracy_mean": float(differences.mean()),
            "embedding_minus_tfidf_accuracy_std": float(differences.std(ddof=1))
            if len(differences) > 1
            else 0.0,
            "per_seed_differences": differences.tolist(),
        }
    retention_gate = any(
        (metrics["committed_pair_accuracy"]["mean"] or 0) >= 0.90
        and (metrics["top_ranked_ticket_coverage"]["mean"] or 0) >= 0.50
        and (metrics["top_ranked_ticket_accuracy"]["mean"] or 0) >= 0.90
        for metrics in binary_summary["fewshot"]["16"].values()
    )
    accuracy_gate = paired_deltas["16"]["embedding_minus_tfidf_accuracy_mean"] >= 0.05
    continue_decision = accuracy_gate and retention_gate

    # Measure the deployment-shaped 16-shot model with seed zero.
    deployment_support = sample_supports(train_labels, categories, shots=16, seed=0)
    deployment_coefficients, deployment_intercepts = fit_heads(
        train_vectors, train_labels, categories, deployment_support
    )
    head_params = {
        category.replace("_", " "): (
            deployment_coefficients[index],
            float(deployment_intercepts[index]),
        )
        for index, category in enumerate(categories)
    }
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
        "fewshot": {
            "runs": results["fewshot"],
            "summary": performance_summary["fewshot"],
        },
        "tfidf_same_shots": {
            "representation": "shared word and bigram TF-IDF vocabulary fitted on the union of sampled support texts; one balanced logistic-regression head per predicate",
            "runs": results["tfidf"],
            "summary": performance_summary["tfidf"],
        },
        "paired_accuracy_differences": paired_deltas,
        "binary_decision_analysis": {
            "decision_rule": "true if score >= 0.5 + margin; false if score < 0.5 - margin; otherwise unknown",
            "margins": list(margins),
            "ticket_resolution": "all 77 decisions known and exactly one true",
            "models": binary_summary,
        },
        "continue_rule": {
            "accuracy": "at 16 shots, embeddings beat paired same-seed TF-IDF by at least 5 percentage points",
            "retention": "at one predeclared margin, committed-pair accuracy >= 90%, top-ranked ticket coverage >= 50%, and accuracy on those tickets >= 90%",
            "accuracy_gate_passed": accuracy_gate,
            "retention_gate_passed": retention_gate,
            "continue": continue_decision,
            "decision": "continue with few-shot embeddings"
            if continue_decision
            else "negative result; do not promote this approach",
        },
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
            "comparison_rule": "full-data result is context only; matched k-shot comparison uses identical support rows and seeds",
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": report["continue_rule"],
                "paired_accuracy_differences": paired_deltas,
                "duckdb": report["duckdb"],
                "report": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
