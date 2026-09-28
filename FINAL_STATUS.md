# SemPred iterative build status

## Champion
V4 hybrid TF-IDF semantic predicate classifier.

- Base dataset: 24,000 examples
- Predicates: 20
- Ordinary held-out F1: 97.77%
- Ordinary AUROC: 99.73%
- Hard benchmark F1: 97.89%
- Hard benchmark AUROC: 99.60%
- Hard benchmark Brier: 0.0285

## Important regression
A newly constructed 12,000-example ultra-stress benchmark intentionally contains compositional negation, historical mentions, context wrappers, and mixed evidence. V4 scored:

- Accuracy: 82.87%
- F1: 82.90%
- AUROC: 84.83%

This is a real failure and is retained in the repository. The evidence-aggregation branch was also tested and rejected (74.22% accuracy).

## What this proves
The semantic-predicate task can be solved very strongly on a controlled benchmark, but neither the classical classifier nor the first pretrained NLI candidate solves it robustly. No checkpoint is promoted to production.

## NLI candidate update (2026-09-28)
An Apache-2.0 `cross-encoder/nli-MiniLM2-L6-H768` checkpoint was downloaded and pinned to revision `c847a3c0e1cad93a5343183ef183f3044e3fc7c2`. A zero-shot run scored 54.48% accuracy / 19.49% F1 on the 12k stress set with the initial wording. Selecting a hypothesis template and fitting calibration on the v6 hard development set raised the later exploratory stress run to 60.08% accuracy / 41.39% F1. This is not a blind test result. The records are in `benchmarks/results/` and the candidate details are in `MODEL_CARD.md`.

Full fine-tuning on adversarial corpora with family-held-out validation is implemented in `training/finetune_nli.py`. Two runs failed the launch gates:

- Fine-tuned v1: 84.51% accuracy on stress; 46.19% on hard.
- Fine-tuned v2 (43,258 training rows): 85.19% on stress; 46.64% on hard.
- Fine-tuned v3 with the original NLI three-class head: 82.22% on stress; 48.99% on hard.
- DeBERTa zero-shot: 62.13% on stress; 78.25% on hard at about 59 rows/sec CPU.
- A classical/NLI blend hit 85.64% stress only after tuning on that same benchmark, and scored 44.85% on hard; this is a development result, not a release result.

## Latest candidate (2026-09-29)
A MIT-licensed DeBERTa-v3-base NLI model was fine-tuned for one epoch using
25,660 training rows from `sempred_32k_adversarial.jsonl`. Its family-held-out
validation reached 99.15% accuracy, but this is not independent evidence: it
uses the same generated source corpus and selects among checkpoints. On the
exploratory 12k stress benchmark it reached 85.16% accuracy (85.36% F1) at
68.5 rows/sec on CPU. On the hard benchmark it reached 49.37% accuracy.

Threshold sweeps showed that the DeBERTa zero-shot model peaks at 78.99%
accuracy on hard cases and 71.31% on stress; threshold adjustment does not
resolve the transfer gap. The new `benchmark.run_nli` report records accuracy,
precision, recall, and F1 across thresholds from 0.05 to 0.95. The user's
current gate is >=85% accuracy on every benchmark. No model meets it, and none
is promoted. The source benchmarks have been used during model development,
so a future candidate also needs a newly authored locked evaluation set.

An exploratory two-checkpoint blend reached 87.49% stress and 78.99% hard
when each benchmark selected its own settings. The best shared blend and
threshold scored 76.16% on the weaker suite, and no shared setting reached
85% on both. Full per-setting results are in `benchmarks/results/`.

## Iterations
V0.1: baseline API + synthetic data
V4: robust 24k dataset + hybrid classifier — champion
V5: adversarial experiments
V6: evidence aggregation — rejected
V7: 12k ultra-stress benchmark — exposes compositional weakness
V8: 90k curriculum attempt — CPU timeout; no result claimed
V9: MIT DeBERTa NLI fine-tune — stress gate reached, hard transfer rejected

## Remaining release gates
1. Resolve hard-case transfer while retaining >=85% accuracy on stress.
2. Build newly authored, locked semantic benchmark examples with unseen
   lexical forms, domains, and compositions; do not tune on that final set.
3. Require >=85% accuracy on every benchmark before promoting a checkpoint.
4. Report calibration, selective coverage, throughput, memory, and cost per
   million decisions on representative hardware.
5. Integrate the production candidate into DuckDB and benchmark 1M+ rows.
