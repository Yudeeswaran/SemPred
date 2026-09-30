"""Evaluate semantic predicate backends as a 77-way Banking77 intent ranker."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from sempred.nli import NLISemPred

DATASET_REVISION = "796a4623935746f71378f0ebd435635a8ce08e50"
DATASET_URL = f"https://huggingface.co/datasets/PolyAI/banking77/tree/{DATASET_REVISION}"
DATASET_SHA256 = "535fc96c4c2b4c2dbdeb0d4b31f24a4859620cf663d6c19c1bd1f97450c410be"
TRAIN_SHA256 = "4526edfa60622ff9b39e238657ab6d712f6aba1ba91c9d7ed7897b0715ee0390"
DATASET_CARD_SHA256 = "e5cdf68e693c02d60dfbbe1b6deb8c4e585a3a72957ff669f3af5fa3fb0bca76"


def read_dataset(path: Path, card_path: Path, *, split: str = "test") -> tuple[list[str], list[str], list[str]]:
    expected_hash = DATASET_SHA256 if split == "test" else TRAIN_SHA256
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
        raise ValueError(f"{split} data hash does not match the pinned Banking77 snapshot")
    card_content = card_path.read_bytes()
    if hashlib.sha256(card_content).hexdigest() != DATASET_CARD_SHA256:
        raise ValueError("dataset card hash does not match the pinned Banking77 snapshot")
    metadata = yaml.safe_load(card_content.decode("utf-8").split("---", 2)[1])
    names = metadata["dataset_info"]["features"][1]["dtype"]["class_label"]["names"]
    label_names = {int(index): name for index, name in names.items()}
    import duckdb

    rows = duckdb.connect().execute(
        "SELECT text, label FROM read_parquet(?)", [str(path.resolve())]
    ).fetchall()
    texts = [text.strip() for text, _ in rows]
    labels = [label_names[int(label)] for _, label in rows]
    if any(not text or not label for text, label in zip(texts, labels)):
        raise ValueError("Banking77 contains an empty text or category")
    categories = sorted(set(labels))
    expected_rows = 3080 if split == "test" else 10003
    if len(categories) != 77 or len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} {split} rows and 77 categories, found {len(rows)} rows / {len(categories)} categories")
    return texts, labels, categories


def rank_with_model(model, texts: list[str], predicates: list[str]) -> tuple[np.ndarray, float]:
    scores = np.empty((len(texts), len(predicates)), dtype=np.float32)
    started = time.perf_counter()
    for column, predicate in enumerate(predicates):
        if column % 10 == 0:
            print(f"scoring predicate {column + 1}/{len(predicates)}", file=sys.stderr, flush=True)
        for start in range(0, len(texts), model.batch_size if hasattr(model, "batch_size") else 512):
            batch = texts[start : start + (model.batch_size if hasattr(model, "batch_size") else 512)]
            predictions = model.predict_batch(batch, predicate)
            scores[start : start + len(batch), column] = [p.probability for p in predictions]
    return scores, time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-card", type=Path, help="dataset card downloaded with fetch_banking77.py")
    parser.add_argument("--limit-per-class", type=int, help="deterministic balanced test slice; omit for the full test split")
    parser.add_argument("--backend", choices=("tfidf", "nli"), required=True)
    parser.add_argument("--mode", choices=("intent", "predicate"), default="intent")
    parser.add_argument("--predicate", help="official Banking77 category name or its space-separated form")
    parser.add_argument("--model", type=Path, help="local NLI model directory")
    parser.add_argument("--train-data", type=Path, help="official Banking77 training Parquet for TF-IDF")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--dynamic-int8", action="store_true", help="quantize Linear layers to int8 on CPU")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.backend == "nli" and not args.model:
        parser.error("--model is required for --backend nli")
    if args.mode == "predicate" and not args.predicate:
        parser.error("--predicate is required for --mode predicate")
    if args.backend == "tfidf" and not args.train_data:
        parser.error("--train-data is required for --backend tfidf")
    if args.dynamic_int8 and args.backend != "nli":
        parser.error("--dynamic-int8 is available only for the NLI backend")
    if args.dynamic_int8 and args.device != "cpu":
        parser.error("--dynamic-int8 requires --device cpu")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.torch_threads < 1:
        parser.error("--torch-threads must be positive")
    if args.limit_per_class is not None and args.limit_per_class < 1:
        parser.error("--limit-per-class must be positive")

    card_path = args.dataset_card or args.dataset.with_name("banking77_README.md")
    texts, labels, categories = read_dataset(args.dataset, card_path)
    if args.limit_per_class is not None:
        selected = []
        for category in categories:
            candidates = [index for index, label in enumerate(labels) if label == category]
            if args.limit_per_class > len(candidates):
                parser.error(f"--limit-per-class exceeds available test rows for {category}")
            candidates.sort(key=lambda index: hashlib.sha256(texts[index].encode("utf-8")).digest())
            selected.extend(candidates[:args.limit_per_class])
        texts = [texts[index] for index in selected]
        labels = [labels[index] for index in selected]
    # Banking77's released class labels supply candidate predicates directly;
    # only underscores are changed to spaces. No task-specific prompt templates.
    predicates = [category.replace("_", " ") for category in categories]
    selected_category = None
    if args.mode == "predicate":
        matches = [
            category for category in categories
            if category.replace("_", " ").casefold() == args.predicate.replace("_", " ").casefold()
        ]
        if len(matches) != 1:
            parser.error("--predicate must match exactly one official Banking77 category")
        selected_category = matches[0]
        predicate = selected_category.replace("_", " ")

    setup_started = time.perf_counter()
    if args.backend == "tfidf":
        train_texts, train_labels, train_categories = read_dataset(args.train_data, card_path, split="train")
        if train_categories != categories:
            raise ValueError("train and test category sets do not match")
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        model = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=200_000),
            LogisticRegression(max_iter=1000, solver="lbfgs"),
        )
        if args.mode == "predicate":
            train_labels = [int(label == selected_category) for label in train_labels]
        model.fit(train_texts, train_labels)
        model_name = (
            "TF-IDF + LogisticRegression (Banking77 train split, one-vs-rest)"
            if args.mode == "predicate"
            else "TF-IDF + LogisticRegression (Banking77 train split)"
        )
    else:
        import torch

        torch.set_num_threads(args.torch_threads)
        model = NLISemPred.load(args.model, device=args.device)
        model.batch_size = args.batch_size
        model.cache_size = 0
        if args.dynamic_int8:
            model.model = torch.ao.quantization.quantize_dynamic(
                model.model, {torch.nn.Linear}, dtype=torch.qint8
            )
            model.model.eval()
        model_name = model.model_id
    setup_seconds = time.perf_counter() - setup_started

    if args.backend == "tfidf":
        started = time.perf_counter()
        probabilities = model.predict_proba(texts)
        inference_seconds = time.perf_counter() - started
        model_classes = list(model.classes_)
        if args.mode == "predicate":
            positive_index = model_classes.index(1)
            probabilities = probabilities[:, positive_index]
            predictions = (probabilities >= 0.5).astype(int)
        else:
            predictions = np.argmax(probabilities, axis=1)
            categories = model_classes
    elif args.mode == "predicate":
        started = time.perf_counter()
        predictions_list = model.predict_batch(texts, predicate)
        inference_seconds = time.perf_counter() - started
        probabilities = np.asarray([prediction.probability for prediction in predictions_list])
        predictions = (probabilities >= model.threshold).astype(int)
    else:
        probabilities, inference_seconds = rank_with_model(model, texts, predicates)
        predictions = np.argmax(probabilities, axis=1)
    if args.mode == "predicate":
        expected = np.asarray([int(label == selected_category) for label in labels])
        predicted_categories = []
        top3_accuracy = None
    else:
        expected = np.asarray([categories.index(label) for label in labels])
        predicted_categories = [categories[index] for index in predictions]
        top_three = np.argsort(-probabilities, axis=1)[:, :3]
        top3_accuracy = float(np.mean([expected[i] in top_three[i] for i in range(len(expected))]))
    pair_count = len(texts) if args.mode == "predicate" else len(texts) * len(categories)
    report = {
        "dataset": "PolyAI Banking77 official test split",
        "dataset_url": DATASET_URL,
        "dataset_revision": DATASET_REVISION,
        "dataset_license": "CC BY 4.0; attribute PolyAI and cite Casanueva et al. 2020",
        "test_dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "training_dataset_sha256": TRAIN_SHA256 if args.backend == "tfidf" else None,
        "examples": len(texts),
        "evaluation_slice": "full test split" if args.limit_per_class is None else f"deterministic {args.limit_per_class} per class",
        "candidate_predicates": 1 if args.mode == "predicate" else len(categories),
        "candidate_predicate_rule": "official category name with underscores replaced by spaces",
        "backend": args.backend,
        "mode": args.mode,
        "predicate": predicate if args.mode == "predicate" else None,
        "model": model_name,
        "model_path": str(args.model) if args.backend == "nli" else None,
        "model_revision": model.revision if args.backend == "nli" else None,
        "dynamic_int8": args.dynamic_int8,
        "torch_threads": args.torch_threads if args.backend == "nli" else None,
        "setup_seconds": setup_seconds,
        "inference_seconds": inference_seconds,
        "candidate_pairs_scored": pair_count if args.backend == "nli" else None,
        "pairs_per_second": pair_count / inference_seconds if args.backend == "nli" else None,
        "queries_per_second": len(texts) / inference_seconds,
        "accuracy": float(accuracy_score(expected, predictions)),
        "top3_accuracy": top3_accuracy,
        "balanced_accuracy": float(balanced_accuracy_score(expected, predictions)),
        "precision": float(precision_score(expected, predictions, zero_division=0)) if args.mode == "predicate" else None,
        "recall": float(recall_score(expected, predictions, zero_division=0)) if args.mode == "predicate" else None,
        "f1": float(f1_score(expected, predictions, zero_division=0)) if args.mode == "predicate" else None,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "per_intent_accuracy": {
            category: float(np.mean(np.asarray(predicted_categories)[np.asarray(labels) == category] == category))
            for category in categories
        } if args.mode == "intent" else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
