# Product definition

## The first product to validate

SemPred should start as a local semantic triage tool for teams that already
query support-ticket data with Python and DuckDB. A data engineer writes a
plain-language condition such as “the customer is asking for a refund,” runs
it against a text column, and gets a score plus `true`, `false`, or `unknown`.
`unknown` rows stay visible for review instead of being silently discarded.

This is a beachhead to validate, not a proven market claim. The project has no
customer pilot or independent ticket benchmark yet. The first customer should
be a software or commerce team with a repeatable, high-volume ticket triage
task and a reason to keep ticket text in its own environment.

## Buyer and job

- **Buyer:** a data or analytics engineering lead who owns a DuckDB/Python
  pipeline over operational text.
- **User:** the engineer defining and monitoring a semantic filter.
- **Job:** find records that meet a changing, contextual condition without
  writing fragile keyword rules or sending every row to a hosted LLM.
- **Initial boundary:** English support-ticket text and low-consequence
  routing, analytics, and review queues. Do not target automated account,
  payment, eligibility, legal, medical, or other high-impact decisions.

DuckDB already supports indexed full-text search. SemPred should not be sold as
another keyword or document-search feature; the proposed distinction is
predicate scoring over records, with an explicit review state. That difference
must be demonstrated against FTS, rules, and a hosted LLM on a real workload.

## MVP offer

Keep the first release developer-focused. A customer should be able to install
the Python package, load a local model, register DuckDB functions, run an
example query on their own ticket table, and inspect uncertain rows. The first
release should include:

1. A reproducible install on supported Python versions and operating systems.
2. A small end-to-end ticket triage example using a local file and DuckDB.
3. A documented predicate, score, and abstention contract, including model
   download size, latency, memory, and offline behavior.
4. A workload-specific evaluation report comparing SemPred with a simple rule,
   an embedding classifier, and one hosted LLM baseline.
5. Clear controls for model provenance, local data handling, failure behavior,
   and version upgrades.

Do not build a web dashboard, hosted service, or broad integration matrix until
a design partner confirms that the DuckDB workflow solves a paid problem.

## Pilot acceptance

Recruit a design partner with permission to use a representative, de-identified
sample. Freeze a human-reviewed evaluation set before model selection. Compare
all alternatives on the same rows, predicates, thresholds, and hardware. Track
precision and recall per predicate, abstention coverage, review time, throughput,
memory, installation friction, and cost per 10,000 rows. Report confidence
intervals and errors by scenario, not only a blended accuracy figure.

The pilot succeeds only if it improves a business metric the partner chooses
(for example, analyst review time or cost per correctly routed ticket) without
missing the partner's quality floor. A generic 85% score across unrelated
synthetic suites is not a substitute for this acceptance test. Keep automation
in suggestion/review mode until false-negative risk is understood.

## Commercial direction

Start with a free developer preview to earn evaluation and feedback. If pilots
confirm recurring value, test a paid offer around production deployment,
private model adaptation, support, and quality monitoring. Do not set a price
or claim savings until the pilot measures the buyer's baseline costs and the
support burden. A model score alone is not a product moat; packaging, reliable
workflow integration, workload evidence, and support are part of the offer.

## Current state

The repository has a Python API, CLI, DuckDB adapter, local model loading, and
CI. It has no validated ticket-domain model, customer-facing triage workflow,
design partner, or independent ticket evaluation. Existing synthetic
benchmarks were exposed during development. SemPred is therefore a research
prototype, not a product ready for sale.

See the [launch plan](LAUNCH_PLAN.md), [evaluation record](EVALUATION.md),
[annotation guide](ANNOTATION_GUIDE.md), and [model card](MODEL_CARD.md).
