# SemPred

SemPred evaluates a natural-language condition against rows of text in Python
or DuckDB. The first use case to validate is local support-ticket triage: score
a condition such as “the customer is requesting a refund” and route uncertain
rows for review.

> **Product status: research prototype.** SemPred's preferred backend is a local
> transformer cross-encoder. The TF-IDF implementation remains available as a
> fast lexical baseline. No current checkpoint reaches 85% accuracy on both
> the project's hard and compositional stress suites, so the model is not yet
> approved for production decisions.

## Install

```bash
python -m pip install "sempred[nli,duckdb]"
```

For local development, clone the repository and run
`python -m pip install -e ".[nli,duckdb]"`. Python 3.10 or newer is required.

## Train and predict

Create a UTF-8 JSONL file with one labeled example per line. Keep a stable
`family` for each behavior and, where possible, a `construction` label for the
outer writing pattern. Validation holds out construction groups first, then
falls back to family groups when the data has little construction diversity:

```json
{"text":"The customer says: I want my money back.","predicate":"customer is requesting a refund","label":1,"family":"refund","construction":"reported_speech"}
{"text":"Earlier, the order arrived on time.","predicate":"customer is requesting a refund","label":0,"family":"refund","construction":"historical"}
```

Download the pinned transformer checkpoint, then fine-tune it on your data:

```bash
sempred download-nli --output models/sempred-nli
sempred train --data labeled.jsonl --base-model models/sempred-nli --model models/sempred-finetuned --epochs 1 --max-length 128
sempred predict --model models/sempred-finetuned --text "Please return my payment" --predicate "customer is requesting a refund"
```

The sample rows only show the format. For the default split, provide at least
three construction groups (or three family groups if no construction groups
are available), with both labels represented in training and validation. If
`construction` is omitted, the trainer recognizes common wrappers and falls
back to `family`; if `family` is also omitted, it groups by exact predicate
text. Use `--validation-group-count` to change how many groups are held out.
This split reduces known template leakage, but it is not a substitute for a
new, independently authored final benchmark. Run `sempred train --help` for
options.

Before fine-tuning generated adversarial data, inspect potential label conflicts:

```bash
python -m training.audit_labels --data labeled.jsonl --output artifacts/label-audit.json
```

The audit flags rows for review; it never changes labels automatically.

The command prints a JSON object with `probability`, `label`, and `decision`.
`unknown` means the score falls inside the configured abstention margin around
the decision threshold. Probabilities are model scores; calibrate them on a
separate representative validation set before interpreting them as confidence.

## Pretrained NLI model

For a real pretrained semantic encoder, install the optional runtime and save
the pinned MIT-licensed model locally:

```bash
python -m pip install "sempred[nli]"
sempred download-nli --output models/sempred-nli --cache-dir .cache/huggingface
sempred predict --model models/sempred-nli --text "I cancelled last month" --predicate "customer wants to cancel a subscription"
sempred calibrate --model models/sempred-nli --output models/sempred-nli-calibrated --data heldout-calibration.jsonl
```

The first command downloads the checkpoint from Hugging Face; inference after
that uses the saved local model. The current default is
[`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli),
pinned to a fixed revision so reruns use the same weights. This pretrained
checkpoint is an evaluation candidate, not a promoted SemPred model. Its
entailment score must pass the release benchmarks before production use.
Calibration data must be held out from training and final evaluation, and must
reflect the workload where scores will be used.

To fine-tune on your labeled corpus, from a source checkout run:

```bash
sempred train --backend nli --data labeled.jsonl --base-model models/sempred-nli --model models/sempred-finetuned --epochs 1
```

Validation holds out construction groups when available and falls back to
complete family groups when it cannot identify enough distinct constructions.
This reduces template leakage but does not prove generalization to unseen
domains. Do not use the final evaluation dataset for fine-tuning or calibration.

## Python API

```python
from sempred import NLISemPred

model = NLISemPred.load("models/sempred-nli")
result = model.predict("I cancelled last month", "customer wants to cancel a subscription")
print(result.probability, result.decision)

batch = model.predict_batch(texts, "customer wants to cancel a subscription")
```

`SemPred.fit(...)` remains available for the lightweight TF-IDF baseline; use
`sempred train --backend tfidf` to select it explicitly. Both backends bound
their in-memory prediction cache (10,000 pairs by default); set
`cache_size=0` when constructing a model to disable it for high-cardinality
streaming workloads.

## DuckDB

```python
import duckdb
from sempred import NLISemPred
from sempred.duckdb import register

model = NLISemPred.load("models/sempred-finetuned")
con = register(duckdb.connect(), model)
rows = con.execute("""
    SELECT ticket_text, SEM_SCORE(ticket_text, 'customer is requesting a refund') AS score
    FROM tickets
    WHERE SEM_PREDICT(ticket_text, 'customer is requesting a refund') IS TRUE
""").fetchall()
```

`SEM_PREDICT` maps `unknown` to SQL `NULL`; `SEM_SCORE` returns the raw model
score. With the `duckdb` extra, SemPred registers Arrow UDFs and scores each
DuckDB chunk through the batch API. Without PyArrow it falls back to scalar
Python UDFs. The vectorized path still needs an end-to-end million-row
benchmark before its throughput can be claimed.

## Independent benchmark

The benchmark uses the official [PolyAI Banking77](https://huggingface.co/datasets/PolyAI/banking77/tree/796a4623935746f71378f0ebd435635a8ce08e50)
test split (3,080 examples, 77 intents), pinned to an immutable dataset
revision. Candidate predicates come directly from the released intent names;
only underscores are replaced with spaces. No SemPred task templates are
written for this evaluation. The TF-IDF classifier is trained on the separate
official 10,003-example training split. The dataset is CC BY 4.0; see its card
for the required PolyAI attribution. Cite Casanueva et al. (2020),
[“Efficient Intent Detection with Dual Sentence Encoders”](https://arxiv.org/abs/2003.04807).

| Baseline | Training / prompt setup | Accuracy | Inference rows/sec | Evaluation |
| --- | --- | ---: | ---: | --- |
| TF-IDF + logistic regression | Banking77 train split, one-vs-rest | 98.73% | 46,417 | Full test split; predicate `card swallowed` |
| Zero-shot MiniLM NLI | `cross-encoder/nli-MiniLM2-L6-H768` | 97.89% | 34.5 | Same binary predicate task |
| Fine-tuned MiniLM NLI | Synthetic SemPred corpus; no Banking77 training | 85.88% | 25.0 | Same binary predicate task |
| Zero-shot MiniLM NLI, dynamic int8 | Same zero-shot checkpoint, CPU int8 | 98.73% | 54.0 | Same task; quality regressed |
| [FLAN-T5-small](https://huggingface.co/google/flan-t5-small) | Pinned zero-shot yes/no prompt | 98.70% | 27.7 | Same binary task; predicted no for every row |

All measured rows use the full 3,080-example test split and a single candidate
predicate, `card swallowed`, taken from the official category list. There are
40 positive examples and 3,040 negatives. Raw accuracy is therefore misleading;
balanced accuracy and F1 are more informative:

| Baseline | Balanced accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| TF-IDF + logistic regression | 51.25% | 1.00 | 0.025 | 0.049 |
| Zero-shot MiniLM NLI | 70.56% | 0.288 | 0.425 | 0.343 |
| Fine-tuned MiniLM NLI | 75.58% | 0.058 | 0.650 | 0.107 |
| Zero-shot MiniLM NLI, dynamic int8 | 51.25% | 1.00 | 0.025 | 0.049 |
| FLAN-T5-small | 50.00% | 0.000 | 0.000 | 0.000 |

The NLI checkpoints were run on CPU with 8 PyTorch threads and batch size 128.
FLAN-T5-small used batch size 64 and the same 8-thread CPU.
Rates include tokenization and forward inference, exclude model loading, and
were measured on a Windows 11 laptop with an Intel `Family 6 Model 154` CPU,
Python 3.13.5, and PyTorch 2.14.0+cpu. TF-IDF inference excludes fit; the
one-vs-rest fit took 0.4 seconds. These are single-machine measurements, not
general performance guarantees. Dynamic int8 increased speed by about 56% but
reduced F1 from 0.343 to 0.049 at the unchanged 0.5 threshold. This row uses
PyTorch's deprecated dynamic-quantization API as an experiment; choose and
revalidate a maintained quantization runtime before shipping int8 inference.

A separate 77-way intent-classification baseline reaches 85.45% top-1 accuracy
and 41,880 rows/sec on the full test split when trained on the official
Banking77 training split. This is a conventional intent classifier; it should
not be confused with the one-predicate results above or SemPred's binary API.

The FLAN-T5 run used `google/flan-t5-small` at revision
`0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab` (Apache-2.0) and a fixed yes/no
prompt. Its 98.70% raw accuracy hides that it answered “no” for every row; its
balanced accuracy and F1 are both zero. The checkpoint weights are verified by
SHA-256 in the runner. FLAN-T5 is an instruction-tuned encoder-decoder model,
included here as a small generative baseline, not a hosted general-purpose
assistant.

To reproduce these runs, install the development extras and fetch the
hash-checked dataset and model files:

```bash
python -m pip install -e ".[dev,nli]"
python benchmark/fetch_banking77.py
python -m benchmark.fetch_flan_t5
python benchmark/banking77.py \
  --dataset .cache/research-data/banking77_test.parquet \
  --train-data .cache/research-data/banking77_train.parquet \
  --backend tfidf \
  --mode predicate --predicate "card swallowed" \
  --output .cache/research-data/banking77-tfidf.json
```

Run the LLM baseline with:

```bash
python -m benchmark.banking77_llm \
  --dataset .cache/research-data/banking77_test.parquet \
  --model-dir .cache/research-data/flan-t5-small \
  --output .cache/research-data/banking77-flan-t5-small.json
```

Replace `--backend tfidf` with `--backend nli --model models/sempred-nli` or
`--model models/sempred-finetuned-v3-neutral` for the NLI rows; add
`--dynamic-int8` to reproduce the quantized row. A 77-way NLI run is also
available by omitting `--mode predicate`, but the full 77-candidate CPU run is
costly. One predicate on one public dataset does not establish quality for
another support workload or make automated decisions safe without customer
data, calibration, and review.

## Model files and privacy

Inference runs locally. Training data and text are not sent to a service by
this package. TF-IDF models are saved as a ZIP archive containing JSON
configuration and NumPy arrays; NumPy loading disables pickle. Older pickle
model files are intentionally rejected and must be retrained. Transformer
checkpoints use Hugging Face safetensors. Keep training examples representative,
and review errors before deploying any filter that might discard important
records.

## Project status and documentation

SemPred is not ready for sale. It has an independent public support-ticket
benchmark, but no customer pilot or customer-specific evaluation. Product
scope, model details, evaluation history, architecture, and reproducibility
notes are in the linked docs. Workload-level validation, calibrated quality
claims, optimized inference, and the million-row DuckDB benchmark remain open.

- [Product scope](docs/PRODUCT.md)
- [Model card](docs/MODEL_CARD.md)
- [Evaluation history](docs/EVALUATION.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Reproducibility](docs/REPRODUCIBILITY.md)

## License

MIT. See [LICENSE](LICENSE).
