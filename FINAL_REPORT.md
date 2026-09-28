# SemPred — Iterative Build Report

## Champion
**V4 Hybrid** — word + character TF-IDF features with calibrated logistic regression.

Why this checkpoint:
- 24,000 training examples generated across 20 semantic predicates.
- 4,320 ordinary held-out examples.
- 240 adversarial examples.
- Hard-set accuracy: **97.92%**.
- Hard-set F1: **97.89%**.
- Hard-set AUROC: **99.60%**.
- Brier score: **0.0285**.
- Ordinary held-out F1: **97.77%**.

V4 Word has the same hard accuracy/F1 and slightly higher hard AUROC, while V4 Hybrid has materially better calibration (lower Brier score) and is therefore the selected semantic-score checkpoint.

## Iterations

| Version | Dataset | Evaluation | Best result | Decision |
|---|---:|---|---:|---|
| V1 | 16k | random split | 99.6% accuracy | Rejected: leakage/template repetition |
| V4 | 24k | adversarial | 97.9% accuracy | **Champion** |
| V5 | 32k | 1,060 adversarial | 69.7% accuracy | Rejected: benchmark exposed whole-document distractor weakness |
| V6 | 22k atomic + 2,680 hard docs | compositional | 85.4% accuracy | Stress-test branch; not promoted |

## Dataset design

The dataset contains semantic predicates covering:
- refunds
- cancellation
- delivery problems
- billing problems
- unauthorized transactions
- PII
- breaking schema changes
- pipeline failures
- overdue payments
- password reset
- software bugs
- sentiment
- security incidents
- chargebacks
- duplicate records
- address changes
- account lock
- plan upgrades
- data export

Training examples include paraphrases, contextual wrappers, lexical substitutions, distractor statements, negation, temporal state, hypothetical language, and role changes.

## Important benchmark rule

The initial 16k benchmark is **not** considered a meaningful quality benchmark because random template splitting produced inflated scores. V4+ uses adversarial/compositional examples specifically to expose this failure mode.

## What remains

This is **not yet a production-quality 100–150M neural SemPred model**. The current champion is a strong local baseline proving the task and benchmark. The next research step is a pretrained encoder/cross-encoder fine-tuned on the 24k+32k corpus, followed by teacher distillation, calibration, quantization, and million-row inference benchmarking.

No claim of perfect accuracy is made. The current benchmark is designed to prevent us from fooling ourselves with template leakage.
