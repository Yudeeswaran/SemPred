# SemPred — Complete Project Guide

## 0. What this repository is

SemPred is an experimental database-native semantic predicate engine.

The core problem is simple:

> Given a piece of text and a natural-language predicate, determine whether the text satisfies the predicate, with a probability and a safe decision.

Example:

```text
Text:
"The customer says the package arrived three days late and wants a refund."

Predicate:
"customer is requesting a refund"

Output:
probability = 0.98...
decision   = true
```

The intended value is scale. Calling a large language model once for every row in a table can be expensive and slow. SemPred aims to put a small, fast, local/open-weight semantic model in the execution path and reserve a stronger teacher for uncertain cases.

This is **not** a chatbot, **not** a generic NL-to-SQL generator, and **not** merely an intent classifier.

---

# 1. The original product idea

The initial design was a semantic decision primitive that could eventually appear inside data systems:

```text
SQL / Python / DataFrame
        |
        v
semantic predicate
        |
        v
candidate pruning
        |
        v
small semantic model
        |
        +---- TRUE
        +---- FALSE
        +---- UNKNOWN -> optional teacher
```

The long-term interfaces are:

- Python API;
- DuckDB function;
- Polars integration;
- Spark integration;
- later warehouse-native integrations.

---

# 2. Why the problem is technically difficult

A keyword classifier can learn that `refund` is associated with `refund requested`.

A real semantic predicate engine must understand cases such as:

```text
Text: "The customer mentioned that her friend got a refund, but she does not want one."
Predicate: "customer is requesting a refund"
```

The word `refund` appears, but the predicate is false.

Other difficult cases include:

- negation;
- historical events;
- quoted speech;
- attribution;
- multiple people/entities;
- temporal qualifiers;
- mixed positive and negative evidence;
- paraphrases;
- unseen lexical forms;
- compositional combinations that were never seen during training.

This is why the project eventually moved away from treating high synthetic benchmark scores as proof of general semantic understanding.

---

# 3. Repository layout

```text
sempred/
├── sempred/                 # Runtime package
│   ├── core.py              # Main SemPred API
│   ├── encoder.py           # Encoder/model abstraction
│   ├── calibration.py       # Calibration utilities
│   ├── duckdb.py             # DuckDB integration surface
│   └── __init__.py
│
├── training/                # Training experiments
│   ├── synthetic.py
│   ├── iterative_build.py
│   ├── v4_robust.py
│   ├── v5_adversarial.py
│   ├── v6_evidence.py
│   ├── v7_stress.py
│   └── v8_curriculum.py
│
├── benchmark/               # Evaluation code
│   ├── metrics.py
│   ├── hard_cases.py
│   ├── run.py
│   └── sellability.py
│
├── data/                    # Generated research datasets
├── artifacts/               # Model/result artifacts
├── tests/                   # Regression tests
├── examples/                # Usage examples
├── docs/                    # Architecture/reproducibility/product docs
├── .github/workflows/       # CI
│
├── FINAL_REPORT.md
├── FINAL_STATUS.md
├── SELLABILITY_STATUS.md
└── COMPLETE_PROJECT_GUIDE.md
```

---

# 4. Version history

## V0.1 — baseline

The first implementation established:

- text + predicate input;
- TF-IDF representation;
- logistic regression classifier;
- probability output;
- Python API;
- synthetic training data;
- basic benchmark machinery.

The key design decision was to keep the public API independent from the model implementation.

That means the eventual transformer/cross-encoder can replace the TF-IDF backend without changing the user-facing interface.

---

# 5. V4 — first strong classical baseline

The V4 line introduced a substantially stronger controlled dataset and hybrid representation.

Recorded results:

- 24,000 examples;
- 20 predicate families;
- ordinary held-out F1 around 97.8%;
- controlled hard-set F1 around 97.9%;
- hard-set AUROC around 99.60%;
- hard-set Brier score around 0.0285.

This became the classical champion.

However, these results were not sufficient to claim that SemPred understood semantics generally.

---

# 6. The critical failure that changed the project

A new 12,000-example ultra-stress benchmark was deliberately constructed with harder semantic compositions:

- negation;
- historical mentions;
- context wrappers;
- mixed evidence;
- compositional combinations;
- wording variations.

The V4 champion dropped to approximately:

```text
Accuracy  82.87%
F1        82.90%
AUROC     84.83%
```

This was retained rather than hidden.

That result is one of the most important artifacts in the repository because it demonstrates that the classical model can exploit lexical/topic correlations without reliably solving compositional semantic reasoning.

---

# 7. V5 — adversarial experiments

The V5 work expanded adversarial examples and hard negatives.

The goal was to answer:

> Does the model still work when superficial lexical cues are intentionally misleading?

Some experiments improved controlled performance, but the broader stress problem remained.

The conclusion was that adding more synthetic templates alone was not enough.

---

# 8. V6 — evidence aggregation

An evidence-aggregation approach was tested.

It looked promising on atomic/controlled examples but failed on the harder compositional benchmark.

Recorded stress behavior was around 74% accuracy for the rejected branch.

Decision:

**Rejected.**

Reason:

The approach added complexity without solving the fundamental semantic generalization problem.

---

# 9. V7 — ultra-stress benchmark

V7 formalized the difficult benchmark as a first-class regression suite.

This changed the development philosophy:

```text
High score on synthetic data
        !=
semantic understanding
```

The stress suite became a release gate rather than an optional experiment.

---

# 10. V8 — curriculum attempt

A larger adversarial curriculum was attempted, targeting roughly 90,000 examples.

The available CPU-only environment could not complete the intended training run within the execution window.

No result was fabricated.

Status:

**Incomplete / blocked by environment.**

---

# 11. Product/runtime work added after the model experiments

The runtime was hardened around the model-independent API.

## Prediction

A prediction contains:

```python
Prediction(
    probability=...,
    label=...,
    decision="true" | "false" | "unknown",
)
```

## Why UNKNOWN exists

For semantic filtering, forcing every case into TRUE/FALSE is dangerous.

The runtime therefore supports an abstention margin:

```text
high probability  -> TRUE
low probability   -> FALSE
middle band       -> UNKNOWN
```

This does not magically improve a bad model. Coverage and quality must be reported together.

---

# 12. Teacher escalation

The intended production architecture allows uncertain predictions to be sent to a stronger teacher.

```text
local model
    |
    +---- confident -> result
    |
    +---- uncertain -> teacher
                         |
                         v
                       cache
```

The cache key is based on the text/predicate pair so repeated questions do not repeatedly invoke the teacher.

The teacher is an optional safety/quality mechanism, not the core economic proposition.

The product is still intended to avoid one expensive LLM call per database row.

---

# 13. Calibration

Raw classifier probabilities are not automatically trustworthy probabilities.

The repository therefore includes calibration infrastructure and treats calibration as a release requirement.

Future model reports should include:

- ECE;
- Brier score;
- reliability behavior;
- selective accuracy vs coverage;
- abstention rate.

---

# 14. Benchmark philosophy

Every model must be evaluated at multiple levels.

## Ordinary set

Measures normal task performance.

## Hard set

Adds harder but still controlled examples.

## Stress set

Designed specifically to defeat lexical shortcuts.

## External set

Must come from an independent public dataset or independently authored evaluation set.

The final model should not be promoted based only on the synthetic SemPred datasets.

---

# 15. Required metrics

Quality:

- accuracy;
- precision;
- recall;
- F1;
- AUROC;
- AUPRC;
- ECE;
- Brier score.

Selective prediction:

- coverage;
- accuracy at coverage;
- UNKNOWN rate.

Systems:

- p50 latency;
- p95 latency;
- throughput;
- memory;
- model size;
- cost per 1M decisions.

Database scale:

- 10K rows;
- 100K rows;
- 1M rows;
- eventually larger fixed benchmarks.

---

# 16. The intended model upgrade

The classical model is not the final model.

The next intended backend is a compact pretrained NLI/semantic cross-encoder.

A candidate investigated during development was:

`cross-encoder/nli-MiniLM2-L6-H768`

The reason for choosing an NLI-style model is that the task has the structure:

```text
premise/text + hypothesis/predicate
                 -> entailment probability
```

The intended training process is:

```text
pretrained NLI model
        ↓
SemPred predicate corpus
        ↓
hard-negative mining
        ↓
compositional curriculum
        ↓
calibration
        ↓
quantization
        ↓
production runtime
```

The development environment used for the current iteration had no cached checkpoint and no working outbound model/package download path. Therefore this model has **not** been falsely represented as trained or benchmarked here.

---

# 17. Teacher distillation direction

Once a stronger teacher is available, the desired pipeline is:

```text
large/strong teacher
        |
        +--> semantic labels
        +--> confidence
        +--> hard negatives
        |
        v
training corpus
        |
        v
small SemPred model
```

The goal is to transfer semantic behavior into a much smaller model.

---

# 18. Candidate pruning

For database-scale operation, evaluating every expensive semantic model against every row may still be wasteful.

The intended runtime therefore supports a two-stage architecture:

```text
all rows
  ↓
cheap candidate pruning
  ↓
small semantic model
  ↓
optional teacher
```

Potential pruning mechanisms:

- metadata filtering;
- partitions;
- lexical retrieval;
- BM25;
- embeddings;
- database statistics.

The pruning layer must preserve recall. It cannot be allowed to silently remove true matches.

---

# 19. DuckDB direction

The target user experience is something like:

```sql
SELECT *
FROM tickets
WHERE SEM_PREDICT(
    ticket_text,
    'customer is requesting a refund'
);
```

The implementation must eventually support vectorized execution instead of making a Python call per row.

The DuckDB gate is not yet complete.

---

# 20. Million-row performance gate

The repository includes a performance target for 1M rows.

The current sparse baseline can process smaller fixed batches at roughly the tens-of-thousands-of-rows-per-second scale in this environment, but the full 1M benchmark did not complete within the available execution window.

This is recorded as an unresolved performance gate.

It is **not** acceptable to extrapolate the smaller result and claim the 1M benchmark passed.

The production benchmark should compare:

```text
baseline
vs
optimized local model
vs
teacher/LLM reference
```

using the same rows, hardware class, and task.

---

# 21. Cost thesis

The product thesis is:

```text
per-row external LLM calls
        ↓
expensive + high latency

small local semantic model
        ↓
cheap + batchable + predictable

uncertain minority
        ↓
strong teacher
```

The actual cost advantage must be measured, not assumed.

---

# 22. What is currently proven

The repository currently demonstrates:

1. the semantic predicate API can be implemented cleanly;
2. controlled datasets can produce very strong classical results;
3. the same model can fail badly under harder compositional tests;
4. tri-state decisions and abstention can be represented at the runtime layer;
5. teacher escalation can be cached;
6. benchmark/reproducibility gates can be encoded as engineering artifacts;
7. the current classical backend is not yet a general semantic model.

---

# 23. What is NOT proven

The repository does not currently prove:

- production-grade semantic understanding;
- >=95% stress accuracy;
- >=98% ordinary/hard accuracy for the final model;
- million-row production throughput;
- optimized neural inference latency;
- a completed DuckDB extension;
- a measured LLM cost advantage on a representative external workload;
- customer willingness to pay.

These are intentionally open gates.

---

# 24. Sellability definition

SemPred can be called a sellable product only after all major gates pass:

### Quality

- >=95% accuracy on the locked compositional stress suite;
- >=98% ordinary/hard quality;
- external benchmark validation;
- calibrated probabilities;
- no critical semantic regression.

### Performance

- 1M+ row benchmark completed;
- optimized/quantized inference;
- predictable memory use;
- strong batch throughput.

### Integration

- usable Python package;
- DuckDB integration;
- documented API;
- clear failure/UNKNOWN semantics.

### Economics

- measured cost per 1M decisions;
- representative comparison to LLM baseline;
- teacher usage small enough that the economics still work.

### Engineering

- CI;
- reproducible benchmark commands;
- versioned datasets;
- model hashes;
- regression tests;
- installation documentation;
- release packaging.

### Product

- clear customer workflow;
- example datasets;
- onboarding;
- error handling;
- licensing documentation;
- customer-facing documentation.

---

# 25. How to continue the project

## Step 1 — install dependencies

```bash
python -m pip install -e '.[dev]'
```

For neural development:

```bash
python -m pip install -e '.[ml]'
```

For DuckDB development:

```bash
python -m pip install -e '.[duckdb]'
```

## Step 2 — run tests

```bash
pytest -q
```

## Step 3 — run the existing demo

```bash
python examples/train_demo.py
```

## Step 4 — inspect the baseline experiments

```bash
python training/v4_robust.py
python training/v5_adversarial.py
python training/v6_evidence.py
python training/v7_stress.py
```

Check the corresponding artifacts in `artifacts/`.

## Step 5 — run the stress benchmark

Use the benchmark scripts and the frozen stress dataset under `data/`.

The stress benchmark is the most important current regression because it caught the classical model's semantic weakness.

## Step 6 — build the pretrained encoder path

The next environment should:

1. obtain a verified pretrained NLI checkpoint;
2. record model ID/license/hash;
3. fine-tune it on the SemPred corpus;
4. generate hard negatives;
5. evaluate on a completely locked test set;
6. calibrate;
7. benchmark inference;
8. only then promote it.

---

# 26. Git/CI workflow

The intended workflow is:

```text
feature branch
    ↓
implementation
    ↓
local tests
    ↓
benchmark
    ↓
commit
    ↓
GitHub CI
    ↓
review
    ↓
merge
```

Every meaningful model change should record:

- why it was made;
- what hypothesis it tests;
- exact configuration;
- metrics;
- failure modes;
- accept/reject decision.

Do not delete failed experiments simply because they are embarrassing. They are part of the evidence.

---

# 27. Current GitHub state

The intended repository is:

`Yudeeswaran/SemPred`

The GitHub repository exists, but during the current ChatGPT session the available GitHub connector allowed repository reads while returning HTTP 403 for write operations. Therefore this source bundle is the authoritative handoff artifact for the work completed in this environment.

The local development commits that were created during the iterative work should be treated as local history until a write-capable GitHub/Codex workflow pushes them.

Do not claim that GitHub contains a commit unless GitHub actually reports it.

---

# 28. Final status at this handoff

```text
PRODUCT STATUS:      Research prototype / product candidate
MODEL STATUS:        Classical baseline; final semantic model not complete
QUALITY GATE:        FAIL — compositional stress ~82.9% accuracy
PERFORMANCE GATE:    OPEN — 1M benchmark incomplete
DUCKDB GATE:         OPEN
EXTERNAL DATA GATE:  OPEN
CALIBRATION GATE:    PARTIAL
CI:                  Included in source bundle
REPRODUCIBILITY:     Documented
SELLABLE:            NO — not yet
```

The most important next action is **not another cosmetic API change**. It is to obtain and train a real compact pretrained semantic/NLI encoder, then make it earn promotion through the locked stress, external, calibration, and million-row gates.

---

# 29. Engineering principle

The project should follow one rule above all others:

> **Never turn an unverified experiment into a product claim.**

If a test fails, keep the failure.
If an environment blocks a test, mark it blocked.
If a benchmark is synthetic, label it synthetic.
If a model has not been downloaded and run, do not report its result.
If a performance number was measured only on 10K rows, do not call it a 1M-row result.

That is how this repository remains useful when another developer or coding agent continues the work.

---

# 30. Continuation update — 2026-09-28

The earlier handoff statement that no checkpoint or network path was available
describes the previous iteration. The current local environment later regained
shell access and downloaded the pinned Apache-2.0
`cross-encoder/nli-MiniLM2-L6-H768` model. The new backend is in `sempred/nli.py`;
the CLI has `download-nli`, `calibrate`, and `predict` workflows; and
`benchmark/run_nli.py` records model and dataset revisions and calibration
metrics.

The first zero-shot run did not improve the product gate: 54.48% accuracy and
19.49% F1 on the 12k compositional benchmark. A later run using a hypothesis
template selected and calibrated on the v6 hard development data scored
60.08% accuracy and 41.39% F1 on that same stress set. This is exploratory
evidence only: stress examples were inspected during this work, so the result
cannot be treated as a blind release evaluation. Both result records and this
limitation are retained under `benchmarks/results/` and `MODEL_CARD.md`.

Full NLI fine-tuning now holds out whole predicate families from the 32k
adversarial corpus for validation. A fine-tuned result still has to meet the
ordinary, hard, compositional, external, calibration, systems, and integration
gates before SemPred can be described as sellable.
