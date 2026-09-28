from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

from sklearn.model_selection import train_test_split

from benchmark.metrics import evaluate
from sempred import SemPred
from training.synthetic import generate


def main():
    rows = generate(n_per_family=1000)
    train, test = train_test_split(rows, test_size=.25, random_state=42, stratify=[r.family for r in rows])
    model = SemPred.fit([r.text for r in train], [r.predicate for r in train], [r.label for r in train])
    start = perf_counter()
    probs = [model.score(r.text, r.predicate) for r in test]
    elapsed = perf_counter() - start
    metrics = evaluate([r.label for r in test], probs)
    metrics.update({
        "examples": len(test),
        "seconds": elapsed,
        "examples_per_second": len(test) / elapsed,
    })
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/benchmark.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
