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
