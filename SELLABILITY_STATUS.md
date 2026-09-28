# SemPred sellability status

## Current verdict: NOT SELLABLE YET

The product skeleton is usable, but the current model does not meet the robust semantic-quality gate.

### Strong results
- 24k base corpus / 20 predicate families.
- Ordinary held-out F1: ~97.8%.
- Controlled hard-set F1: ~97.9%.
- 12k ultra-stress benchmark: retained as the primary regression gate.
- Local sparse baseline throughput: ~10.5k rows/sec on the 12k stress batch in this environment.

### Blocking result
The classical champion scores 82.87% accuracy / 82.90% F1 / 84.83% AUROC on
the 12k ultra-stress set. The best exploratory stress blend currently reaches
85.64%, but its weight was selected on that same set and the blend scores only
44.85% on hard cases. A stronger DeBERTa checkpoint reaches 78.25% on hard but
62.13% on stress. No current candidate clears >=85% on both suites without
benchmark-specific tuning; none is promotable.

This means the current model is learning lexical/topic correlations rather than reliably representing compositional semantics.

### Product work completed in this iteration
- Added tri-state decision support: `true`, `false`, `unknown`.
- Added selective-coverage benchmark.
- Added sellability benchmark artifact.
- Preserved all previous datasets and checkpoints.
- Added a `sempred` CLI to train from labeled JSONL and emit single-pair JSON predictions.
- Added DuckDB `SEM_SCORE` and tri-state `SEM_PREDICT` functions (`unknown` becomes SQL `NULL`).
- Bounded the local prediction cache and documented trusted-only pickle loading.
- Rewrote the README around install, train, Python, and DuckDB workflows.
- Added a pinned pretrained NLI integration, calibration, and family-held-out fine-tuning path.
- Added six automated core, calibration, and DuckDB tests and verified that all pass.
- Built a distributable Python wheel successfully.
- Added model comparison tooling and tested an MIT-licensed DeBERTa NLI checkpoint.

The product workflow is easier to try, but these changes do not affect the
model's semantic-quality score or close the sellability gates below.

### Required before selling
1. A reliable cross-encoder candidate that passes independent validation; current candidates fail.
2. 10k+ newly authored, locked semantic stress cases with unseen wording and compositions.
3. >=95% accuracy on the stress gate and >=98% on ordinary/hard evaluation.
4. Workload-representative calibration and selective prediction.
5. ONNX/INT8 inference benchmark and 1M+ row DuckDB throughput measurement.
6. Demonstrated cost/latency advantage against a representative LLM baseline.
7. Customer-facing workflow, reproducible installation, and external validation.

No model is promoted to production merely because it performs well on a leaked/template-heavy split.
