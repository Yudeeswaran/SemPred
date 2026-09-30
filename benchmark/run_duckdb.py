"""Benchmark unique-pair NLI evaluation through the Arrow-backed DuckDB UDF."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import duckdb
import pyarrow as pa

from sempred.duckdb import register
from sempred.nli import NLISemPred


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="saved local SemPred NLI model directory")
    parser.add_argument("--dataset", type=Path, required=True, help="local labeled JSONL source corpus")
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.rows < 1:
        parser.error("--rows must be positive")

    model_started = time.perf_counter()
    model = NLISemPred.load(args.model, device=args.device)
    model.cache_size = 0  # measure unique-pair inference without cache hits
    model_load_seconds = time.perf_counter() - model_started

    content = args.dataset.read_bytes()
    source = [json.loads(line) for line in content.decode("utf-8").splitlines() if line.strip()]
    if not source:
        raise ValueError("benchmark source dataset is empty")
    generation_started = time.perf_counter()
    texts = [f"Record {index}: {source[index % len(source)]['text']}" for index in range(args.rows)]
    predicates = [source[index % len(source)]["predicate"] for index in range(args.rows)]
    data = pa.table({"text": texts, "predicate": predicates})
    generation_seconds = time.perf_counter() - generation_started

    con = register(duckdb.connect(), model)
    con.register("sempred_benchmark_input", data)
    query_started = time.perf_counter()
    result = con.execute(
        "WITH scored AS MATERIALIZED ("
        "SELECT SEM_PREDICT(text, predicate) AS decision "
        "FROM sempred_benchmark_input) "
        "SELECT count(*) AS rows, "
        "sum(CASE WHEN decision IS TRUE THEN 1 ELSE 0 END) AS true_rows, "
        "sum(CASE WHEN decision IS NULL THEN 1 ELSE 0 END) AS unknown_rows "
        "FROM scored"
    ).fetchone()
    query_seconds = time.perf_counter() - query_started
    con.close()

    report = {
        "model": model.model_id,
        "revision": model.revision,
        "dataset": str(args.dataset),
        "dataset_sha256": hashlib.sha256(content).hexdigest(),
        "rows": result[0],
        "true_rows": result[1],
        "unknown_rows": result[2],
        "device": args.device,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "duckdb": duckdb.__version__,
        "pyarrow": pa.__version__,
        "model_load_seconds": model_load_seconds,
        "input_build_seconds": generation_seconds,
        "query_seconds": query_seconds,
        "rows_per_second": result[0] / query_seconds,
        "cache_size": model.cache_size,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
