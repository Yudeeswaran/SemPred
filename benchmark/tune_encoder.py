"""Compare CPU batching and optional ONNX inference on Banking77 text."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

from benchmark.banking77 import DATASET_REVISION, read_dataset
from sempred.fewshot import (
    DEFAULT_ENCODER_ID,
    DEFAULT_ENCODER_REVISION,
    FrozenTextEncoder,
)


def _numpy_mean_pool(hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
    weights = mask.astype(np.float32)[..., None]
    pooled = (hidden * weights).sum(axis=1) / np.maximum(weights.sum(axis=1), 1e-9)
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    return pooled / np.maximum(norms, 1e-12)


def _torch_measure(
    encoder, texts: list[str], batch_size: int, threads: int, *, sort_by_length: bool
) -> dict:
    encoder.torch.set_num_threads(threads)
    encoder.batch_size = batch_size
    encoder.sort_by_length = sort_by_length
    encoder.encode(texts[: min(batch_size, len(texts))])
    started = time.perf_counter()
    vectors = encoder.encode(texts)
    seconds = time.perf_counter() - started
    return {
        "seconds": seconds,
        "texts_per_second": len(texts) / seconds,
        "texts": len(vectors),
    }


def _export_onnx(encoder, destination: Path) -> None:
    torch = encoder.torch

    class HiddenState(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            ).last_hidden_state

    sample = encoder.tokenizer(
        ["short support message", "a slightly longer support message"],
        padding=True,
        truncation=True,
        max_length=encoder.max_length,
        return_tensors="pt",
    )
    wrapper = HiddenState(encoder.model).eval()
    torch.onnx.export(
        wrapper,
        (sample["input_ids"], sample["attention_mask"], sample["token_type_ids"]),
        str(destination),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "token_type_ids": {0: "batch", 1: "sequence"},
            "last_hidden_state": {0: "batch", 1: "sequence"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )


def _onnx_measure(session, tokenizer, texts: list[str], batch_size: int) -> dict:
    vectors = []
    started = time.perf_counter()
    for start in range(0, len(texts), batch_size):
        tokens = tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="np",
        )
        inputs = {
            name: tokens[name]
            for name in ("input_ids", "attention_mask", "token_type_ids")
        }
        hidden = session.run(["last_hidden_state"], inputs)[0]
        vectors.append(_numpy_mean_pool(hidden, tokens["attention_mask"]))
    seconds = time.perf_counter() - started
    return {"seconds": seconds, "texts_per_second": len(texts) / seconds}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-card", type=Path, required=True)
    parser.add_argument("--encoder", type=Path)
    parser.add_argument("--sample-size", type=int, default=512)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[32, 64, 128, 256]
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--onnx-path",
        type=Path,
        default=Path(".cache/research-data/minilm-encoder.onnx"),
    )
    args = parser.parse_args()
    if args.sample_size < 32 or any(
        value < 1 for value in args.threads + args.batch_sizes
    ):
        parser.error(
            "sample-size must be >= 32 and thread/batch sizes must be positive"
        )

    texts, _, _ = read_dataset(args.dataset, args.dataset_card)
    rng = np.random.default_rng(77)
    selected = rng.choice(len(texts), min(args.sample_size, len(texts)), replace=False)
    sample = [texts[index] for index in selected]
    encoder_path = args.encoder or DEFAULT_ENCODER_ID
    import torch

    torch.set_num_threads(8)
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

    token_lengths = encoder.tokenizer(
        sample,
        truncation=True,
        max_length=encoder.max_length,
        add_special_tokens=True,
        return_length=True,
    )["length"]
    original_order = sample
    length_order = [text for _, text in sorted(zip(token_lengths, sample))]
    torch_results = []
    for threads in args.threads:
        for batch_size in args.batch_sizes:
            for order_name, ordered_texts in (
                ("sampled", original_order),
                ("length_sorted", length_order),
                ("sort_in_encoder", original_order),
            ):
                sort_by_length = order_name == "sort_in_encoder"
                result = _torch_measure(
                    encoder,
                    ordered_texts,
                    batch_size,
                    threads,
                    sort_by_length=sort_by_length,
                )
                torch_results.append(
                    {
                        "runtime": "torch",
                        "threads": threads,
                        "batch_size": batch_size,
                        "order": order_name,
                        "sort_by_length": sort_by_length,
                        **result,
                    }
                )

    onnx_result: dict
    try:
        import onnx
        import onnxruntime as ort

        args.onnx_path.parent.mkdir(parents=True, exist_ok=True)
        if not args.onnx_path.is_file():
            _export_onnx(encoder, args.onnx_path)
        onnx_model = onnx.load(str(args.onnx_path))
        onnx.checker.check_model(onnx_model)
        parity_options = ort.SessionOptions()
        parity_options.intra_op_num_threads = 1
        parity_options.inter_op_num_threads = 1
        parity_session = ort.InferenceSession(
            str(args.onnx_path), parity_options, providers=["CPUExecutionProvider"]
        )
        parity_texts = sample[: min(32, len(sample))]
        parity_tokens = encoder.tokenizer(
            parity_texts,
            padding=True,
            truncation=True,
            max_length=encoder.max_length,
            return_tensors="np",
        )
        parity_inputs = {
            name: parity_tokens[name]
            for name in ("input_ids", "attention_mask", "token_type_ids")
        }
        onnx_hidden = parity_session.run(["last_hidden_state"], parity_inputs)[0]
        onnx_vectors = _numpy_mean_pool(onnx_hidden, parity_tokens["attention_mask"])
        encoder.batch_size = len(parity_texts)
        torch_vectors = encoder.encode(parity_texts)
        cosine = np.sum(torch_vectors * onnx_vectors, axis=1)
        parity = {
            "examples": len(parity_texts),
            "cosine_similarity_mean": float(cosine.mean()),
            "cosine_similarity_min": float(cosine.min()),
            "max_absolute_embedding_difference": float(
                np.max(np.abs(torch_vectors - onnx_vectors))
            ),
        }
        onnx_results = []
        for threads in args.threads:
            options = ort.SessionOptions()
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            session = ort.InferenceSession(
                str(args.onnx_path), options, providers=["CPUExecutionProvider"]
            )
            for batch_size in args.batch_sizes:
                for order_name, ordered_texts in (
                    ("sampled", original_order),
                    ("length_sorted", length_order),
                ):
                    result = _onnx_measure(
                        session, encoder.tokenizer, ordered_texts, batch_size
                    )
                    onnx_results.append(
                        {
                            "runtime": "onnxruntime",
                            "threads": threads,
                            "batch_size": batch_size,
                            "order": order_name,
                            **result,
                        }
                    )
        onnx_result = {
            "available": True,
            "version": ort.__version__,
            "path": str(args.onnx_path),
            "model_sha256": hashlib.sha256(args.onnx_path.read_bytes()).hexdigest(),
            "torch_parity": parity,
            "results": onnx_results,
        }
    except ImportError as exc:
        onnx_result = {"available": False, "reason": str(exc)}

    report = {
        "dataset": "PolyAI Banking77 official test split",
        "dataset_revision": DATASET_REVISION,
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "encoder": {
            "model": DEFAULT_ENCODER_ID,
            "revision": DEFAULT_ENCODER_REVISION,
            "max_length": encoder.max_length,
            "sample_size": len(sample),
            "token_length_mean": float(np.mean(token_lengths)),
            "token_length_p95": float(np.percentile(token_lengths, 95)),
        },
        "torch_results": torch_results,
        "onnx": onnx_result,
        "hardware": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "torch": torch.__version__,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    best_torch = max(torch_results, key=lambda row: row["texts_per_second"])
    best_onnx = (
        max(onnx_result["results"], key=lambda row: row["texts_per_second"])
        if onnx_result.get("available")
        else None
    )
    print(
        json.dumps(
            {
                "best_torch": best_torch,
                "best_onnx": best_onnx,
                "report": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
