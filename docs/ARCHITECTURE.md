# SemPred Architecture

## Problem

SemPred evaluates a natural-language predicate against text at database/batch scale. It is a semantic decision primitive, not a chatbot and not a general NL-to-SQL system.

Example:

```text
Text:      "The package arrived three days late and the customer wants a refund."
Predicate: "customer is requesting a refund"
Result:    probability + TRUE/FALSE/UNKNOWN decision
```

## Target architecture

```text
SQL / Python / DataFrame
          |
          v
+-----------------------+
| Query / runtime layer |
+-----------+-----------+
            |
            v
+-----------------------+
| Candidate pruning     |
| metadata / lexical    |
| / partition filters   |
+-----------+-----------+
            |
            v
+-----------------------+
| Compact semantic      |
| predicate model       |
+-----------+-----------+
            |
            v
 probability + decision
     |          |
     |          +--> UNKNOWN --> optional teacher --> cache
     v
 TRUE/FALSE
```

## Current implementation

The checked-in baseline is a TF-IDF + logistic-regression cross-input classifier. It deliberately exposes a stable API so the model backend can be replaced without rewriting integrations.

The baseline is a research control, not the final semantic model.

## Intended production model

The next model is a compact pretrained NLI/semantic cross-encoder, fine-tuned on predicate examples and hard negatives. The current environment did not have a pretrained checkpoint or network access, so no unverified model is included.

## Runtime contract

The model must provide:

- `score(text, predicate) -> probability`
- `predict(text, predicate) -> Prediction`
- `predict_batch(texts, predicate)`
- stable serialization
- deterministic inference for a fixed model/configuration

## Tri-state semantics

A probability threshold alone is unsafe for semantic decisions. SemPred therefore supports:

- `true`: sufficiently above the decision threshold
- `false`: sufficiently below the decision threshold
- `unknown`: inside the configured abstention margin

UNKNOWN is a product feature, not a mechanism for hiding poor accuracy. Release benchmarks must report both coverage and quality.

## Teacher escalation

For uncertain cases, an optional stronger teacher can answer the predicate. Teacher answers are cached using a stable hash of `(text, predicate)` to avoid repeated calls.

The teacher path is optional. The product goal remains a fast local/open-weight model for the common case.

## Database integration

The intended first native integration is DuckDB, with a scalar predicate function and vectorized/batched execution. Later adapters can target Polars, Spark, and cloud warehouses.

## Performance strategy

Performance comes from:

1. batching;
2. candidate pruning;
3. compact model inference;
4. quantization/ONNX or equivalent optimized runtime;
5. avoiding one external LLM request per row;
6. optional teacher escalation only for uncertain rows.
