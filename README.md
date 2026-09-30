# SemPred

SemPred evaluates natural-language predicates against support-ticket text in
Python or DuckDB. It returns a score and a tri-state decision: `true`, `false`,
or `unknown` when the score is too close to the threshold. `unknown` maps to
SQL `NULL` in DuckDB so uncertain rows can be routed for review.

**Status: research prototype, not ready for production use.** The current
few-shot experiment is fast after embedding, but misses the accuracy of the
full-data TF-IDF baseline on Banking77. No backend is promoted as a dependable
customer-facing classifier.

## Current experiment

The active hypothesis is a frozen `all-MiniLM-L6-v2` sentence encoder plus a
small balanced logistic-regression head for each predicate. For each input
text, the encoder is called once; candidate predicate heads operate on the
cached vector. The encoder weights are pinned to an immutable revision and
head archives use JSON and NumPy data, not pickle.

The experiment uses the official [PolyAI Banking77](https://huggingface.co/datasets/PolyAI/banking77/tree/796a4623935746f71378f0ebd435635a8ce08e50)
train and test splits. Each predicate receives a balanced support set sampled
from the 10,003-row train split. Both low-shot methods use the identical support
indices for each of 10 seeds and are evaluated on the untouched 3,080-ticket,
77-intent test split. No templates or generated examples are used.

| Method | Labels per predicate | Full-test accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| TF-IDF, one head per predicate | 8 | 47.90% ± 1.85 pp | 47.50% ± 1.64 pp |
| Frozen MiniLM, one head per predicate | 8 | 62.70% ± 2.30 pp | 61.96% ± 2.48 pp |
| TF-IDF, one head per predicate | 16 | 60.25% ± 0.62 pp | 60.37% ± 0.58 pp |
| Frozen MiniLM, one head per predicate | 16 | 72.16% ± 1.92 pp | 71.87% ± 2.07 pp |
| TF-IDF, 77-class full-data baseline | All 10,003 train rows | 85.45% | — |

With the same 16 labels per predicate, frozen embeddings beat TF-IDF by
**11.91 percentage points** on average (paired seed standard deviation: 2.01
points). That is a real label-efficiency gain. It still misses the full-data
baseline, and the predicate-level abstention results do not meet the ticket
coverage bar below.

At the default `0.10` abstention margin, 16-shot embeddings committed 44.1% of
ticket-predicate pairs; 96.3% of those decisions matched the binary labels,
with 38.5% precision, 79.1% recall, and 51.8% F1. Unknown positive pairs count
as unrecalled. At ticket level, selecting the top-scoring predicate committed
88.5% of tickets at 76.7% accuracy. Across the predeclared margins, no point
reached both 90% ticket accuracy and 50% ticket coverage: at margin `0.15`,
coverage was 57.7% and accuracy was 84.6%; at `0.20`, accuracy was 90.2% but
coverage fell to 21.1%. No ticket had all 77 decisions known with exactly one
positive predicate.

For `card swallowed`, 16-shot embeddings at margin `0.10` had 59.5% precision,
81.0% recall, and 67.1% F1, with 25.4% pair coverage. The matched low-shot
TF-IDF head scored 35.7% F1 at the same margin. Earlier single-predicate
cross-encoder measurements were 0.34 F1 zero-shot, 0.11 after synthetic-data
fine-tuning, and 0.05 with int8; those older runs had no abstention and are
included only for context.

On the documented Intel Family 6 Model 154 CPU, PyTorch encoded all 13,072
unique train/test texts in 18.24 seconds (717 texts/sec) with batch size 32,
8 threads, and length sorting. The DuckDB Arrow UDF scored 237,160 pairs across
all 77 predicates in 5.57 seconds (42,582 pairs/sec, or 553 unique ticket
texts/sec), embedding each of the 3,079 unique test texts once. At this measured
rate, one million unique ticket texts extrapolate to about 30 minutes for the
77-predicate query; this is a linear estimate from Banking77, not a service
capacity guarantee. The encoder-only ONNX benchmark reached 864 texts/sec.
DuckDB returned `NULL` for 133,070 low-confidence pairs at the default margin.

The predeclared continuation rule required both a 5-point 16-shot win over
matched TF-IDF and, at one predeclared margin, 90% committed-pair accuracy,
50% confident ticket coverage, and 90% accuracy on those tickets. The accuracy
gate passed; the ticket-retention gate failed. **Do not promote this approach**
without a better abstention/coverage tradeoff.

These are machine-specific measurements on Python 3.13.5, Windows 11, and 8
PyTorch threads. The dataset is CC BY 4.0; retain the PolyAI attribution and cite Casanueva et al. (2020),
[“Efficient Intent Detection with Dual Sentence Encoders”](https://arxiv.org/abs/2003.04807).
See the per-seed quality results in
[`benchmark/results/banking77-fewshot.json`](benchmark/results/banking77-fewshot.json)
and the batch/thread/ONNX measurements in
[`benchmark/results/banking77-encoder-tuning.json`](benchmark/results/banking77-encoder-tuning.json)
and [`benchmark/results/banking77-encoder-thread-sweep.json`](benchmark/results/banking77-encoder-thread-sweep.json).

## Reproduce

Install the benchmark/runtime dependencies and fetch the hash-verified dataset:

```bash
python -m pip install -e ".[embeddings,duckdb,dev]"
python benchmark/fetch_banking77.py
python -c 'from sempred import FrozenTextEncoder; FrozenTextEncoder.download("models/all-MiniLM-L6-v2")'
python -m benchmark.banking77_fewshot \
  --train-data .cache/research-data/banking77_train.parquet \
  --test-data .cache/research-data/banking77_test.parquet \
  --dataset-card .cache/research-data/banking77_README.md \
  --encoder models/all-MiniLM-L6-v2 \
  --seeds 10 \
  --output benchmark/results/banking77-fewshot.json
```

The pinned encoder is `sentence-transformers/all-MiniLM-L6-v2` at revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. The benchmark stores its derived
vectors under the ignored `.cache/` directory for repeat runs; dataset files,
models, and customer corpora are not committed.

## Python and DuckDB API

Use real, labeled examples from your own training split. Every predicate needs
both positive and negative examples. Keep final evaluation and calibration
examples out of the support set.

```python
from sempred import FewShotSemPred, FrozenTextEncoder
from sempred.duckdb import register
import duckdb

FrozenTextEncoder.download("models/all-MiniLM-L6-v2")
encoder = FrozenTextEncoder("models/all-MiniLM-L6-v2")
model = FewShotSemPred.fit(
    texts=real_labeled_texts,
    predicates=predicate_names,
    labels=binary_labels,
    encoder=encoder,
    abstain_margin=0.1,
)
model.save("models/customer-support.zip")

con = register(duckdb.connect(), model)
rows = con.execute("""
    SELECT ticket_text,
           SEM_SCORE(ticket_text, 'refund request') AS score,
           SEM_PREDICT(ticket_text, 'refund request') AS decision
    FROM tickets
""").fetchall()
```

`predict_many(texts, predicates)` batches mixed predicates and deduplicates
text embeddings. The in-memory embedding cache is bounded (10,000 unique texts
by default); set `cache_size=0` to disable it. `SEM_PREDICT` maps `unknown` to
SQL `NULL`. Install the `duckdb` extra to enable Arrow UDFs.

## Project scope

SemPred is not validated for customer workloads or high-impact decisions. The
one-predicate NLI and small-LLM numbers in the history are binary evaluations
and cannot be compared with the 77-way intent results above. Synthetic-data
generation and its demo benchmarks have been removed; historical synthetic
results remain labeled as development-only evidence in
[`docs/EVALUATION.md`](docs/EVALUATION.md).

See [product scope](docs/PRODUCT.md), [evaluation history](docs/EVALUATION.md),
and the [model card](docs/MODEL_CARD.md).
