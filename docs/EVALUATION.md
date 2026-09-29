# Evaluation record

## How to read these results

The original corpora were produced by a synthetic generator with a small set
of predicate families and repeated writing patterns. Scores on random splits
of those corpora mostly measure whether a model learned the generator. The
older development suites were also inspected while selecting model
checkpoints, templates, and thresholds. Treat every score below as historical
debugging evidence, not an independent estimate of production quality.

## Findings so far

| Candidate | Evaluation | Accuracy | Interpretation |
| --- | --- | ---: | --- |
| TF-IDF/logistic regression | 12k compositional stress | 82.87% | Retained failure; below the 85% target |
| TF-IDF/logistic regression | 2,680 hard cases | 97.92% | Controlled set result; not evidence of broad semantic transfer |
| DeBERTa fine-tune v1 | synthetic family validation | 99.15% | Same generated source; not independent evidence |
| DeBERTa fine-tune v1 | 12k stress | 85.16% | Exploratory; suite was already exposed during development |
| DeBERTa fine-tune v1 | 2,680 hard cases | 49.37% | Clear domain/composition transfer failure |
| DeBERTa zero-shot | 2,680 hard cases | 78.25% | Useful reference point; below target |
| DeBERTa zero-shot | 12k stress | 62.13% | Useful reference point; below target |

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
