# SemPred

Evaluate a natural language predicate against text locally, from Python or DuckDB.
SemPred returns a probability and a three-way decision (`true`, `false`, or
`unknown`) so a caller can choose how to handle uncertain cases.

> **Model status: developer preview.** SemPred's preferred backend is a local
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

## Model files and privacy

Inference runs locally. Training data and text are not sent to a service by
this package. Model files use Python pickle: **only load model files you trust**.
Keep training examples representative, and review errors before deploying any
filter that might discard important records.

## Project status and documentation

SemPred is a usable developer preview, not yet a sellable semantic decision
product. The locked stress and external-data gates, calibrated quality claims,
optimized inference, and million-row DuckDB benchmark remain open. See
[the product definition and release gates](docs/PRODUCT.md),
[the model card](docs/MODEL_CARD.md),
[the evaluation history](docs/EVALUATION.md),
[architecture](docs/ARCHITECTURE.md), and
[reproducibility rules](docs/REPRODUCIBILITY.md).

## License

MIT. See [LICENSE](LICENSE).
