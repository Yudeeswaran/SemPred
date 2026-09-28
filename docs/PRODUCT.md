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
from sempred import SemPred

model = SemPred.fit(texts, predicates, labels)
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

The SQL syntax is a product target; the current baseline is not yet the finished DuckDB extension.

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

The Python package now includes an installable command line for training from
labeled JSONL, single-pair prediction, a bounded in-memory cache, and optional
DuckDB score/tri-state functions. With PyArrow the DuckDB adapter uses Arrow
chunks and the batch API; without it the adapter falls back to scalar Python
UDFs. The vectorized path still needs an end-to-end million-row benchmark
before its throughput can be claimed. The TF-IDF model artifact uses Python
pickle and must only be loaded from a trusted source; downloaded NLI weights
use safe-tensor files.

The included classifier remains a TF-IDF/logistic-regression research baseline.
Its 82.87% accuracy on the retained compositional stress set is below the
required threshold. The CLI and integrations make the prototype easier to try;
they do not change that quality result or qualify the engine for production
semantic filtering.

## Release checklist

Do not describe the system as a sellable semantic decision product until every
gate above has reproducible evidence attached to a tagged release. In
particular, a genuine pretrained semantic model and independently authored
evaluation set are prerequisites; synthetic training data alone cannot close
the semantic-quality gate.
