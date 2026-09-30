# Product scope

SemPred evaluates a natural-language predicate against a text value from
Python or DuckDB. The API returns a score and one of three decisions:
`true`, `false`, or `unknown`. The initial use case under evaluation is
low-consequence filtering and triage of English support-ticket text. Unknown
records remain available for review.

The latest all-intent Banking77 experiment found that a 16-example frozen
embedding head reached 72.16% accuracy, below the full-data TF-IDF baseline at
85.45%. Neither that approach nor the older NLI cross-encoder is approved as
the product model. See the [few-shot evaluation](EVALUATION.md#frozen-embedding-few-shot-hypothesis).

## Current interfaces

- Python API for single and batched predictions.
- CLI for model download, training, prediction, and calibration.
- DuckDB functions for semantic score and tri-state predicate queries.
- Local model loading; text is not sent to an inference service by the package.

## Not supported as a product claim

SemPred is not validated as a general-purpose classifier, document search
engine, or automated decision system. A public Banking77 benchmark is
available, but there is no customer-specific evaluation or pilot. Do not use
it to make high-impact decisions.

## Release requirements

Before a production release, the project needs workload-specific, independently
reviewed quality results; calibration and abstention coverage; a measured
comparison with the customer's existing rules and an LLM baseline; clean
installation and upgrade checks; privacy and model-license review; and
end-to-end DuckDB latency and throughput on documented hardware. Do not use the
old synthetic development suites as independent release evidence.

See [architecture](ARCHITECTURE.md), [evaluation](EVALUATION.md),
[reproducibility](REPRODUCIBILITY.md), and the [model card](MODEL_CARD.md).
