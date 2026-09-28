# SemPred

Evaluate a natural language predicate against text locally, from Python or DuckDB.
SemPred returns a probability and a three-way decision (`true`, `false`, or
`unknown`) so a caller can choose how to handle uncertain cases.

> **Model status: research baseline.** The shipped TF-IDF model is not a
> general-purpose semantic reasoner. It scored 82.87% accuracy on the project's
> compositional stress benchmark, below the documented release threshold.
> Treat output as a ranking signal and validate it on your own labeled data
> before using it for consequential decisions.

## Install

```bash
python -m pip install "sempred[duckdb]"
```

For local development, clone the repository and run
`python -m pip install -e ".[duckdb]"`. Python 3.10 or newer is required.

## Train and predict

Create a UTF-8 JSONL file with one labeled example per line:

```json
{"text":"The order arrived late and I want my money back.","predicate":"customer is requesting a refund","label":1}
{"text":"The order arrived on time.","predicate":"customer is requesting a refund","label":0}
```

Include examples from both classes for every behavior you want the model to
recognize. Then train and query a local model:

```bash
sempred train --data labeled.jsonl --model models/refund.pkl --abstain-margin 0.1
sempred predict --model models/refund.pkl --text "Please return my payment" --predicate "customer is requesting a refund"
```

The command prints a JSON object with `probability`, `label`, and `decision`.
`unknown` means the score falls inside the configured abstention margin around
the decision threshold. Probabilities are model scores; calibrate them on a
separate representative validation set before interpreting them as confidence.

## Pretrained NLI model

For a real pretrained semantic encoder, install the optional runtime and save
the pinned Apache-2.0 model locally:

```bash
python -m pip install "sempred[nli]"
sempred download-nli --output models/sempred-nli --cache-dir .cache/huggingface
sempred predict --model models/sempred-nli --text "I cancelled last month" --predicate "customer wants to cancel a subscription"
sempred calibrate --model models/sempred-nli --output models/sempred-nli-calibrated --data heldout-calibration.jsonl
```

The first command downloads the checkpoint from Hugging Face; inference after
that uses the saved local model. The current default is
[`cross-encoder/nli-MiniLM2-L6-H768`](https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768),
pinned to a fixed revision so reruns use the same weights. This pretrained
checkpoint is an evaluation candidate, not a promoted SemPred model. Its
entailment score must pass the release benchmarks before production use.
Calibration data must be held out from training and final evaluation, and must
reflect the workload where scores will be used.

To fine-tune on your labeled corpus, from a source checkout run:

```bash
python -m training.finetune_nli --train-data labeled.jsonl --base-model models/sempred-nli --output models/sempred-finetuned --epochs 2
```

This groups the validation split by `family` to prevent the same predicate
family from appearing in both train and validation data. Supply `family` on
every training row; do not use the final evaluation dataset for fine-tuning or
calibration.

## Python API

```python
from sempred import NLISemPred

model = NLISemPred.load("models/sempred-nli")
result = model.predict("I cancelled last month", "customer wants to cancel a subscription")
print(result.probability, result.decision)

batch = model.predict_batch(texts, "customer wants to cancel a subscription")
```

`SemPred.fit(...)` remains available for the lightweight TF-IDF baseline. Both
backends bound their in-memory prediction cache (10,000 pairs by default); set
`cache_size=0` when constructing a model to disable it for high-cardinality
streaming workloads.

## DuckDB

```python
import duckdb
from sempred import SemPred
from sempred.duckdb import register

model = SemPred.load("models/refund.pkl")
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
[the full project guide](COMPLETE_PROJECT_GUIDE.md),
[product definition and release gates](docs/PRODUCT.md), and
[reproducibility rules](docs/REPRODUCIBILITY.md).

## License

MIT. See [LICENSE](LICENSE).
