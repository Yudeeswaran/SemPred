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
train and test splits. Each predicate gets a balanced support set sampled from
the training split. Results below are 10 support-sampling seeds over the full
3,080-ticket, 77-intent test set. No templates or generated examples are used.
The full-data TF-IDF result is the existing 77-class baseline, trained on all
10,003 official training examples.

| Method | Labeled examples per predicate | Full-test accuracy | Macro F1 | Inference throughput |
| --- | ---: | ---: | ---: | ---: |
| TF-IDF + logistic regression | Full train split | 85.45% | — | 41,880 tickets/sec |
| Frozen MiniLM + predicate heads | 8 (4 positive, 4 negative) | 62.70% ± 2.30 pp | 61.96% ± 2.48 pp | 610k tickets/sec across all 77 heads, cached vectors |
| Frozen MiniLM + predicate heads | 16 (8 positive, 8 negative) | 72.16% ± 1.92 pp | 71.87% ± 2.07 pp | 550k tickets/sec across all 77 heads, cached vectors |

The 16-shot result is **13.29 percentage points below** the TF-IDF baseline,
so the few-shot model does not meet the accuracy criterion and is not a
replacement. On the documented CPU, the encoder processed 13,072 unique
training and test texts at 77 texts/sec. The DuckDB Arrow UDF scored 237,160
ticket-predicate pairs (all 77 predicates) at 4,953 pairs/sec end to end,
embedding each of 3,079 unique test texts once. The warm head-only rate is not
the end-to-end rate. DuckDB returned `NULL` for 133,070 low-confidence pairs
at the default abstention margin. Ranking accuracy uses the highest-scoring
of 77 heads before abstention; multi-class accuracy and binary abstention
coverage are separate metrics.

These are machine-specific measurements on Python 3.13.5, Windows 11, an Intel
Family 6 Model 154 CPU, and 8 PyTorch threads. The dataset is CC BY 4.0; retain
the PolyAI attribution and cite Casanueva et al. (2020),
[“Efficient Intent Detection with Dual Sentence Encoders”](https://arxiv.org/abs/2003.04807).
See the raw runs and pinned data hashes in
[`benchmark/results/banking77-fewshot.json`](benchmark/results/banking77-fewshot.json).

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
