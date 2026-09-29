# Launch plan

## Goal

Sell a local semantic triage workflow that proves its value on one customer's
support-ticket data. The initial product is the Python/DuckDB developer tool;
the first milestone is a paid or explicitly sponsored pilot, not a broad public
launch.

## Sequence

### 1. Confirm the job

Use customer discovery to verify that support teams have a repeated ticket
filtering or routing task that is expensive to maintain with keyword rules.
Record ticket volume, current precision/recall, review effort, data-location
constraints, and the existing alternative. If the buyer has no cost or time
problem, change the wedge before adding product surface.

### 2. Make the pilot easy to try

Ship a small sample project that reads a local CSV or Parquet file, loads the
pinned model, runs one DuckDB query, and writes decisions and scores to an
output file. Include a “review unknowns” path and explicit resource estimates.
Keep raw text local by default. Add an opt-in, documented path for sharing
redacted errors with the project.

### 3. Establish product evidence

Collect separate training, calibration, and final-evaluation data from the
partner's workload, with permission and a written labeling policy. Keep the
final set inaccessible during model development. Compare against the partner's
current rules, a frozen embedding classifier, and a hosted LLM. Use the same
records and prompt/predicate for every baseline. Measure per-class quality,
coverage, latency, throughput, memory, and fully loaded cost. The signed-off
quality and ROI floors come from the pilot buyer; do not tune to the final set.

### 4. Harden the release

Before a public or paid release, provide a clean install on the documented
platforms, pinned dependency and model provenance, schema/input validation,
stable CLI/API behavior, user-facing errors, security/privacy documentation,
license review, reproducible benchmark instructions, and support/upgrade
policy. Verify wheel installation in a clean environment and benchmark the
actual DuckDB path on the intended hardware.

### 5. Convert pilot to offer

If the pilot meets its agreed quality and ROI goals, package the workflow,
publish the evidence with its limits, and propose a paid continuation for
deployment help, private tuning, and support. If quality fails, keep the
product in review mode and use adjudicated errors to improve the training set.
If ROI fails, stop or change the use case rather than adding features to mask it.

## Evidence already available

- Python/CLI/DuckDB developer preview and automated CI.
- Honest exploratory failures retained in the repository.
- An embeddings-plus-logistic-regression pipeline check on an exposed
  synthetic stress suite; it is not product-quality evidence.
- Annotation and evaluation locking guidance.

## Still needed before taking payment

- A validated customer problem and named design partner.
- Permissioned, representative data and an independent adjudicated test set.
- A credible baseline comparison, including a hosted LLM on the same workload.
- A pilot result that meets the customer's quality and cost/time thresholds.
- A clean end-to-end install and documented support boundary.
