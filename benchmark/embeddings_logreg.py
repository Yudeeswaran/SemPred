"""Compare a frozen sentence encoder plus logistic regression with NLI models."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from benchmark.metrics import evaluate

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "8b3219a92973c328a8e22fadcfa821b5dc75636a"


def read_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for number, row in enumerate(rows, 1):
        if not isinstance(row, dict) or not {"text", "predicate", "label"} <= row.keys():
            raise ValueError(f"{path}: row {number} needs text, predicate, and label fields")
        if row["label"] not in (0, 1, False, True):
            raise ValueError(f"{path}: row {number} label must be binary")
    if not rows or len({int(row["label"]) for row in rows}) != 2:
        raise ValueError(f"{path}: expected non-empty data containing both labels")
    return rows


def encode_texts(texts: list[str], *, model_id: str, revision: str, batch_size: int) -> np.ndarray:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install the optional NLI dependencies: pip install 'sempred[nli]'") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModel.from_pretrained(model_id, revision=revision, use_safetensors=True)
    model.eval()
    vectors = []
    for offset in range(0, len(texts), batch_size):
        tokens = tokenizer(texts[offset : offset + batch_size], padding=True, truncation=True,
                            max_length=256, return_tensors="pt")
        with torch.inference_mode():
            output = model(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).to(output.dtype)
            pooled = (output * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        vectors.append(pooled.cpu().numpy())
    return np.concatenate(vectors, axis=0)


def pair_features(text_vectors: np.ndarray, predicate_vectors: np.ndarray) -> np.ndarray:
    return np.concatenate((text_vectors, predicate_vectors,
                           np.abs(text_vectors - predicate_vectors),
                           text_vectors * predicate_vectors), axis=1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", action="append", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True, help="locked evaluation JSONL")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    train = [row for path in args.train_data for row in read_jsonl(path)]
    evaluation = read_jsonl(args.dataset)
    rows = train + evaluation
    unique_texts = list(dict.fromkeys(row["text"] for row in rows))
    unique_predicates = list(dict.fromkeys(row["predicate"] for row in rows))
    started = time.perf_counter()
    all_vectors = encode_texts(unique_texts + unique_predicates, model_id=MODEL_ID,
                               revision=MODEL_REVISION, batch_size=args.batch_size)
    elapsed = time.perf_counter() - started
    text_vector = dict(zip(unique_texts, all_vectors[: len(unique_texts)]))
    predicate_vector = dict(zip(unique_predicates, all_vectors[len(unique_texts) :]))

    def features(part: list[dict]) -> np.ndarray:
        return pair_features(
            np.stack([text_vector[row["text"]] for row in part]),
            np.stack([predicate_vector[row["predicate"]] for row in part]),
        )

    classifier = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    classifier.fit(features(train), [int(row["label"]) for row in train])
    probabilities = classifier.predict_proba(features(evaluation))[:, 1]
    labels = [int(row["label"]) for row in evaluation]
    result = {
        "baseline": "frozen_sentence_embeddings_plus_logistic_regression",
        "encoder": MODEL_ID,
        "encoder_revision": MODEL_REVISION,
        "feature_layout": "text,predicate,abs_difference,elementwise_product",
        "classifier": "sklearn.LogisticRegression(C=1.0,max_iter=1000,random_state=42)",
        "train_examples": len(train),
        "evaluation_examples": len(evaluation),
        "evaluation_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "fit_and_encoding_seconds": elapsed,
        "metrics": evaluate(labels, probabilities),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
