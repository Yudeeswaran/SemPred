# Evaluation record

## How to read these results

The original corpora were produced by a synthetic generator with a small set
of predicate families and repeated writing patterns. Scores on random splits
of those corpora mostly measure whether a model learned the generator. The
older development suites were also inspected while selecting model
checkpoints, templates, and thresholds. Treat every score below as historical
debugging evidence, not an independent estimate of production quality.

## Findings so far

### Independent public dataset

The official PolyAI Banking77 test split is pinned and evaluated separately
from the synthetic SemPred suites. On a single-predicate task using the
released `card swallowed` category, zero-shot MiniLM reached 0.34 F1 at 34.5
rows/sec; the synthetic-corpus fine-tune reached 0.11 F1 at 25.0 rows/sec.

Those figures are binary results for one predicate and must not be compared
with the multi-class results below. The one-predicate FLAN-T5 and NLI numbers
in the original README are retained in the raw benchmark JSON files only.

### Frozen-embedding few-shot hypothesis

This experiment tested whether frozen `sentence-transformers/all-MiniLM-L6-v2`
embeddings plus a small balanced logistic-regression head per intent could
approach the 77-class TF-IDF baseline with 16 labeled examples per intent. The
official 10,003-example train split supplied support sets; the official
3,080-example test split was not used for fitting or tuning. For each of ten
seeds, each intent received four or eight positive examples and the same
number of negative examples sampled from other intents. Head fitting and model
settings were fixed before evaluating the test split.

#### Follow-up decision rule

This follow-up compares both models on the identical support examples for each
of ten seeds at 8 and 16 examples per predicate. Continue with embeddings only
if (a) their 16-shot mean 77-way accuracy beats same-seed TF-IDF by at least
5 percentage points, and (b) at a single predeclared abstention margin from
`0.05`, `0.10`, `0.15`, `0.20`, or `0.25`, committed-pair accuracy is at least
90%, at least 50% of tickets have a confident top-ranked predicate, and
accuracy on those committed tickets is at least 90%. The highest-scoring
predicate is committed only when its score is above the positive threshold
plus the margin. The margin curve is reported as an operating-point analysis;
no margin is tuned after inspecting the held-out results. A stricter
supplementary metric counts tickets as fully resolved only when all 77
decisions are known and exactly one is `true`.

At each margin, committed-pair precision/recall/F1 exclude `unknown` decisions
from positive predictions while positive unknowns count as unrecalled. Ticket
coverage and exact-label accuracy are separate from pair metrics. `card
swallowed` is also reported as a binary one-predicate task for continuity with
the earlier NLI measurements.

| Method | Labeled examples per intent | Accuracy mean ± SD | Macro F1 mean ± SD | Fit all 77 heads |
| --- | ---: | ---: | ---: | ---: |
| TF-IDF, one head per predicate | 8 | 47.90% ± 1.85 pp | 47.50% ± 1.64 pp | 0.11 sec |
| Frozen MiniLM, one head per predicate | 8 | 62.70% ± 2.30 pp | 61.96% ± 2.48 pp | 0.07 sec |
| TF-IDF, one head per predicate | 16 | 60.25% ± 0.62 pp | 60.37% ± 0.58 pp | 0.12 sec |
| Frozen MiniLM, one head per predicate | 16 | 72.16% ± 1.92 pp | 71.87% ± 2.07 pp | 0.08 sec |
| TF-IDF, 77-class full-data baseline | All 10,003 train rows | 85.45% | — | 15.08 sec |

On the paired seeds, frozen embeddings beat same-shot TF-IDF by 14.80
percentage points at 8 examples and 11.91 points at 16 (paired SD: 2.51 and
2.01 points). This is a real low-label advantage. The 16-shot result remains
13.29 points below full-data TF-IDF.

At 16 shots and the default `0.10` margin, pair coverage was 44.1% with 96.3%
accuracy on committed pairs, 38.5% precision, 79.1% recall, and 51.8% F1.
Unknown positive pairs count as unrecalled. Ticket-level top-predicate
coverage was 88.5% at 76.7% accuracy. At margin `0.15`, ticket coverage was
57.7% at 84.6% accuracy; at `0.20`, accuracy reached 90.2% but coverage fell
to 21.1%. No predeclared margin met both ticket gates. No test ticket had all
77 predicates known with exactly one positive decision.

On the one-predicate `card swallowed` task, the 16-shot embedding head at
margin `0.10` averaged 59.5% precision, 81.0% recall, and 67.1% F1 over seeds,
with 25.4% pair coverage. The same-support TF-IDF heads averaged 97.3%
precision, 22.2% recall, and 35.7% F1. The earlier single-predicate NLI scores
of 0.34, 0.11, and 0.05 are included only as historical context; they used
different model/training setups and no abstention.

The measured CPU path used `all-MiniLM-L6-v2`, batch 32, 8 PyTorch threads,
and length-sorted batches. It encoded 13,072 unique train/test texts in 18.24
seconds (717 texts/sec). DuckDB scored 237,160 pairs across all 77 predicates
in 5.57 seconds (42,582 pairs/sec; 553 unique ticket texts/sec), encoding each
of 3,079 unique test texts once. ONNX Runtime reached 864 texts/sec on the full
test split, with mean embedding cosine similarity 1.0 to PyTorch and maximum
absolute difference 2.1e-7. At the measured DuckDB input rate, a million
unique tickets extrapolate to roughly 30 minutes; this is a linear estimate
from this public dataset, not a production capacity guarantee. The full CPU
batch/thread sweep is linked below.

The accuracy gate passed, but the ticket-retention gate failed. The conclusion
is a **negative product result**: this approach shows useful label efficiency
and throughput, but does not meet the agreed ticket-accuracy/coverage bar and
is not promoted.

The benchmark and full per-seed report are in
[`benchmark/banking77_fewshot.py`](../benchmark/banking77_fewshot.py) and
[`benchmark/results/banking77-fewshot.json`](../benchmark/results/banking77-fewshot.json).
Encoder tuning is recorded in
[`benchmark/results/banking77-encoder-tuning.json`](../benchmark/results/banking77-encoder-tuning.json)
and [`benchmark/results/banking77-encoder-thread-sweep.json`](../benchmark/results/banking77-encoder-thread-sweep.json).
Its data and embedding cache are local-only; exact data hashes and the encoder
revision are recorded in the reports.
Dynamic int8 increased speed to 54.0 rows/sec but reduced F1 to 0.05 at the
unchanged threshold. FLAN-T5-small answered no for every example: 27.7
rows/sec, with 0.00 F1. The predicate has only 40 positive test examples among
3,080, so raw accuracy is not a useful quality summary. A Banking77-trained
TF-IDF one-vs-rest model scored 0.05 F1 at 46,417 rows/sec. The conventional
77-class TF-IDF classifier reached 85.45% top-1 accuracy. Full metrics,
commands, hardware details, and dataset revision are in the
[README](../README.md#independent-benchmark).

These measurements are one-predicate and one-dataset evidence, not a claim of
general support-ticket quality. The generative baseline is pinned FLAN-T5-small,
not a hosted frontier LLM. Customer-specific evaluation, calibration, and a
multi-predicate benchmark remain necessary before production claims.

| Candidate | Evaluation | Accuracy | Interpretation |
| --- | --- | ---: | --- |
| TF-IDF/logistic regression | 12k compositional stress | 82.87% | Retained failure; below the 85% target |
| TF-IDF/logistic regression | 2,680 hard cases | 97.92% | Controlled set result; not evidence of broad semantic transfer |
| DeBERTa fine-tune v1 | synthetic family validation | 99.15% | Same generated source; not independent evidence |
| DeBERTa fine-tune v1 | 12k stress | 85.16% | Exploratory; suite was already exposed during development |
| DeBERTa fine-tune v1 | 2,680 hard cases | 49.37% | Clear domain/composition transfer failure |
| DeBERTa zero-shot | 2,680 hard cases | 78.25% | Useful reference point; below target |
| DeBERTa zero-shot | 12k stress | 62.13% | Useful reference point; below target |
| Frozen MiniLM NLI embeddings + logistic regression | same 12k stress | 64.48% | Fixed threshold; trained on 32k synthetic rows; exploratory only |

The embedding baseline used the locally cached NLI MiniLM checkpoint as a
frozen sentence encoder, with independent mean-pooled text and predicate
vectors. It is not the separate sentence-transformers checkpoint described in
the baseline runner's default configuration. Its result is close to the
zero-shot DeBERTa figure on this exposed synthetic stress set and does not
establish that dense embeddings solve the transfer problem. Full provenance,
metrics, and hashes are in
[`../benchmark/results/embeddings_logreg_stress_exploratory.json`](../benchmark/results/embeddings_logreg_stress_exploratory.json).
An external LLM score is not recorded: no provider credential or local general
LLM was available in this workspace, and the pinned sentence-transformer
download was blocked by the workspace network policy.

Threshold and ensemble sweeps selected different settings for different
development suites. Their best per-suite scores are therefore not a valid
shared release result. The JSON files in [`../benchmark/results`](../benchmark/results/)
retain the details and dataset hashes.

## Next evaluation protocol

Create a new, fixed evaluation set with varied domains, predicates, speakers,
time references, negation, and compositional cases. Record its provenance,
labeling rules, class balance, and SHA-256 before scoring any candidate. Keep it
out of training, calibration, and model selection. Compare frozen candidates
against a zero-shot NLI model and a frozen-encoder embedding plus logistic
regression baseline. If a requested external LLM baseline cannot be run with
available credentials or service access, record that limitation instead of
implying a comparison was completed.

New examples produced with assistant help must be labeled as such. They are
not a substitute for independent domain-expert annotation. A single evaluation
on this set is exploratory until its labels and sampling design receive
independent review.

When a locked dataset is available, score it once per frozen candidate. The
zero-shot run uses `python -m benchmark.run_nli`; the dense baseline uses
`python -m benchmark.embeddings_logreg`. Keep output JSON under `artifacts/`
and publish only dataset hashes and summary metrics. The embedding model is
the Apache-2.0 `sentence-transformers/all-MiniLM-L6-v2` checkpoint, pinned to
revision `8b3219a92973c328a8e22fadcfa821b5dc75636a`. Its official model card
describes it as a sentence encoder and provides the mean-pooling procedure
used by the baseline. [Model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).

The next evaluation should be workload-specific. See the
protocol below. No new 500-example gold set has been created yet.

## Collect and lock the next set

Choose one customer workflow and its exact predicates before collecting data.
For the initial support-ticket use case, label whether the text supports the
predicate for the relevant actor and time; keyword overlap alone is not enough.
Mark ambiguous or conflicting rows for adjudication rather than forcing a
binary label. Track scenario group, source type, and annotator for each row.

Sample realistic paraphrases, negation, completed versus requested actions,
past versus current state, hypotheticals, quotations, other speakers, multiple
events, and missing evidence. Avoid multiplying a small number of sentences
through repeated wrappers. Group paraphrases and related scenarios before
splitting so near-duplicates cannot cross partitions. Keep the final set out
of training, calibration, and model selection.

Save the exact JSONL, labeling instructions, source description, split
manifest, and SHA-256. Record annotator agreement and unresolved rows. Score
each frozen candidate once at a predeclared threshold; report per-predicate
quality, calibration, coverage, uncertainty intervals, and error examples. Any
change after inspecting final-set results requires a fresh test set. Assistant-
generated examples can test software but are not independent human evidence.
