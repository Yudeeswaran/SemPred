# Product Definition

## Product statement

SemPred is a local/open-weight semantic predicate engine for evaluating natural-language conditions over large datasets.

## Who it is for

- data engineers;
- analytics/AI platform teams;
- applications that need semantic filtering over large text columns;
- batch classification pipelines where per-row LLM calls are too slow or expensive.

## Example API

```python
from sempred import NLISemPred

model = NLISemPred.load("models/sempred-finetuned")
result = model.predict(
    "The package arrived three days late and the customer wants a refund.",
    "customer is requesting a refund",
)

print(result.probability, result.decision)
```

## Example SQL direction

```sql
SELECT *
FROM support_tickets
WHERE SEM_PREDICT(ticket_text, 'customer is requesting a refund');
```

The package exposes these SQL functions through its DuckDB adapter. Large-scale
latency and one-million-row throughput remain to be measured.

## Release gates

A sellable release requires all of these:

1. semantic quality: >=85% accuracy on every benchmark suite, including
   compositional stress and hard cases;
2. independent locked evaluation and external validation;
3. calibration and selective prediction reported;
4. 1M+ row benchmark completed;
5. optimized/quantized inference benchmark completed;
6. DuckDB end-to-end integration tested;
7. reproducible installation and CI;
8. representative cost/latency comparison against an external LLM baseline;
9. no known critical correctness failure hidden by the benchmark design.

The current repository does not pass all gates.

## Current developer preview

The Python package includes a transformer-first command line for fine-tuning
labeled JSONL, single-pair prediction, a bounded in-memory cache, and optional
DuckDB score/tri-state functions. With PyArrow, the DuckDB adapter uses Arrow
chunks and the batch API; without it, the adapter uses scalar Python UDFs. The
vectorized path still needs an end-to-end million-row benchmark. Downloaded NLI
weights use safe-tensor files. The legacy TF-IDF model uses Python pickle and
should only be loaded from a trusted source.

The preferred runtime and training workflow use a pretrained NLI
cross-encoder. The TF-IDF/logistic-regression implementation remains an
explicit fast baseline; its 82.87% accuracy on the retained compositional
stress set is below the release gate. The NLI fine-tune has not met 85% on every
suite either, so neither backend is approved for production filtering.

## Release checklist

Do not describe the system as a sellable semantic decision product until every
gate above has reproducible evidence attached to a tagged release. In
particular, a genuine pretrained semantic model and independently authored
evaluation set are prerequisites; synthetic training data alone cannot close
the semantic-quality gate.
