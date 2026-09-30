"""Benchmark a pinned local FLAN-T5-small yes/no baseline on Banking77."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from benchmark.banking77 import DATASET_REVISION, read_dataset

MODEL_ID = "google/flan-t5-small"
MODEL_REVISION = "0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab"
MODEL_SHA256 = "495fa51e204676f1a857a9fc13c4c89f3f5ba9f480b898cebca02add25e6d749"
PREDICATE = "card swallowed"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-card", type=Path)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1 or args.torch_threads < 1:
        parser.error("batch size and PyTorch thread count must be positive")

    card_path = args.dataset_card or args.dataset.with_name("banking77_README.md")
    texts, labels, _ = read_dataset(args.dataset, card_path)
    expected = np.asarray([int(label.replace("_", " ").casefold() == PREDICATE) for label in labels])
    weight_path = args.model_dir / "model.safetensors"
    weight_hash = hashlib.sha256(weight_path.read_bytes()).hexdigest()
    if weight_hash != MODEL_SHA256:
        raise ValueError(f"FLAN-T5-small weights do not match pinned revision: {weight_hash}")

    torch.set_num_threads(args.torch_threads)
    setup_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_dir, local_files_only=True, use_safetensors=True
    ).eval()
    model.to("cpu")
    setup_seconds = time.perf_counter() - setup_started

    prompts = [
        "Answer only YES or NO. Does this customer support message describe the intent "
        f"'{PREDICATE}'? Message: {text}"
        for text in texts
    ]
    answers: list[int | None] = []
    inference_started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(prompts), args.batch_size):
            batch = tokenizer(
                prompts[start : start + args.batch_size],
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            generated = model.generate(**batch, max_new_tokens=3, num_beams=1, do_sample=False)
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            for answer in decoded:
                match = re.search(r"\b(yes|no)\b", answer.casefold())
                answers.append(1 if match and match.group(1) == "yes" else 0 if match else None)
    inference_seconds = time.perf_counter() - inference_started

    known = np.asarray([value is not None for value in answers])
    predicted = np.asarray([value or 0 for value in answers])
    known_expected = expected[known]
    known_predicted = predicted[known]
    report = {
        "dataset": "PolyAI Banking77 official test split",
        "dataset_revision": DATASET_REVISION,
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_license": "Apache-2.0",
        "model_weights_sha256": weight_hash,
        "predicate": PREDICATE,
        "prompt": "Answer only YES or NO. Does this customer support message describe the intent '<predicate>'? Message: <text>",
        "examples": len(texts),
        "positive_examples": int(expected.sum()),
        "batch_size": args.batch_size,
        "torch_threads": args.torch_threads,
        "setup_seconds": setup_seconds,
        "inference_seconds": inference_seconds,
        "rows_per_second": len(texts) / inference_seconds,
        "accuracy": float(accuracy_score(known_expected, known_predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(known_expected, known_predicted)),
        "precision": float(precision_score(known_expected, known_predicted, zero_division=0)),
        "recall": float(recall_score(known_expected, known_predicted, zero_division=0)),
        "f1": float(f1_score(known_expected, known_predicted, zero_division=0)),
        "unparsed_rate": float(1.0 - known.mean()),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
