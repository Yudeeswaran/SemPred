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
The user-set quality gate is at least 85% accuracy on every benchmark. The
classical champion scores 82.87% accuracy on the 12k stress suite. A DeBERTa
fine-tune reaches 85.16% on stress but only 49.37% on the hard suite. The
zero-shot DeBERTa reaches 78.25% at the default threshold on hard cases and
62.13% on stress; threshold sweeps peak at 78.99% and 71.31%, respectively.
No current candidate clears 85% across both suites. The fine-tune's 99.15%
family-holdout validation is from the same generated source corpus and is not
independent evidence. None is promotable.

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
3. >=85% accuracy on every benchmark, including hard and stress suites, as
   requested by the product owner.
4. Workload-representative calibration and selective prediction.
5. ONNX/INT8 inference benchmark and 1M+ row DuckDB throughput measurement.
6. Demonstrated cost/latency advantage against a representative LLM baseline.
7. Customer-facing workflow, reproducible installation, and external validation.

No model is promoted to production merely because it performs well on a leaked/template-heavy split.
