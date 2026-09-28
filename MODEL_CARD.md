# SemPred NLI model candidate

## Status

This is an experimental checkpoint, not a production-approved model. The
published stress score is below SemPred's release target, and the stress set
has already been inspected during development. Do not use the reported score
as a blind release result.

## Base checkpoint

- Model: [`cross-encoder/nli-MiniLM2-L6-H768`](https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768)
- Revision: `c847a3c0e1cad93a5343183ef183f3044e3fc7c2`
- License reported by the model repository: Apache-2.0
- NLI sources listed by the repository: SNLI and MultiNLI
- Input: premise is the text; hypothesis is the natural-language predicate
- Default predicate wording: `It is true that {predicate}.`
- Positive score: softmax probability assigned to the NLI `entailment` class

The SemPred wrapper uses an immutable model revision, checks the checkpoint's
label names, applies bounded batching, and stores models in safe-tensor format.
The model files are downloaded separately and are not included in this source
repository.

### Larger NLI experiment

- Model: [`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli)
- Revision: `6f5cf0a2b59cabb106aca4c287eed12e357e90eb`
- License reported by the model repository: MIT
- Training sources listed by the repository: MultiNLI, FEVER, and ANLI
- Size: approximately 184 million parameters
- Checkpoint is local under `models/` and ignored by Git.

## Exploratory results

These numbers record model-selection work; they are not release evidence.

| Run | Dataset | Accuracy | F1 | AUROC | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| Initial zero-shot | 12k stress | 54.48% | 19.49% | 64.21% | First predicate template, uncalibrated |
| Predicate calibration | 12k stress | 60.08% | 41.39% | 72.66% | Template was selected on the v6 hard set; Platt calibration was fit on that same development set |
| Fine-tuned v1, family holdout | 6,340 validation examples | 89.64% | 90.33% | 97.12% | Four predicate families held out during training; validation-informed model selection |
| Fine-tuned v1, stress | 12k stress | 84.51% | 84.46% | 85.51% | Exploratory benchmark; not a locked blind release run |
| Fine-tuned v1, hard development | 2,680 hard examples | 46.19% | 45.83% | 53.00% | Severe domain/composition transfer failure |
| Fine-tuned v2, family holdout | 6,340 validation examples | 89.37% | 90.00% | 96.25% | Retrained on 43,258 rows; validation-informed model selection |
| Fine-tuned v2, stress | 12k stress | 85.19% | 85.18% | 87.32% | Exploratory benchmark; under 95% gate |
| Fine-tuned v2, hard development | 2,680 hard examples | 46.64% | 46.32% | 57.41% | Severe transfer failure remains |
| Fine-tuned v3 neutral, family holdout | 6,340 validation examples | 88.09% | 88.61% | 96.81% | Preserved three-class head; false labels mapped to NLI neutral |
| Fine-tuned v3 neutral, stress | 12k stress | 82.22% | 82.38% | 86.90% | Exploratory; failed 85% target |
| Fine-tuned v3 neutral, hard development | 2,680 hard examples | 48.99% | 50.95% | 58.28% | Exploratory; failed 85% target |
| DeBERTa NLI, zero-shot hard | 2,680 hard examples | 78.25% | 61.97% | 81.79% | MIT checkpoint; 50.6 examples/sec CPU |
| DeBERTa NLI, zero-shot stress | 12k stress | 62.13% | 47.93% | 77.30% | MIT checkpoint; 58.8 examples/sec CPU |

Per-run JSON includes the dataset SHA-256, checkpoint revision, runtime
versions, timing, quality metrics, and selective-coverage measurements under
[`benchmarks/results`](benchmarks/results/). The v6 hard data was used for
prompt selection and calibration, so its fitted metrics must not be presented
as independent evaluation.

Fine-tuned v1 improved substantially over zero-shot on the stress suite, but
still misses the 95% release threshold and failed badly on the v6 hard set.
Family-holdout scores are validation results, not independent external evidence.

The second run added another corpus but did not fix transfer: stress accuracy
was 85.19%, while hard-set accuracy was 46.64%. The training harness now also
supports mapping SemPred negatives to the pretrained NLI `neutral` class,
because many false predicate pairs are not explicit contradictions. This is
an exploratory training strategy; it still needs independent validation.

An arithmetic blend of the classical hybrid and fine-tuned v3 neutral model
reached 85.64% accuracy on stress at an NLI blend weight of 0.4. That weight
was selected on the stress set itself and the same blend reached only 44.85%
on the hard development set; it is not a production result. The stronger
MIT-licensed DeBERTa model improved zero-shot hard accuracy to 78.25% but fell
to 62.13% on stress and was much slower on CPU. A DeBERTa fine-tune is underway.

## Intended use and limits

The candidate is intended to score short English text against a natural-language
claim. It is not validated for long documents, multilingual inputs, legal,
medical, financial, or other high-impact decisions. A high score is not a
calibrated probability until calibrated on a separate workload-representative
set. Applications should review `unknown` cases and measure false negatives
before using SemPred to discard records.

## Promotion requirements

Promotion requires a newly authored locked evaluation set, external validation,
at least 95% accuracy on compositional stress cases, at least 98% ordinary and
hard quality, representative calibration and coverage results, optimized
inference measurements, and the one-million-row database benchmark. Existing
exploratory results do not meet these conditions.
