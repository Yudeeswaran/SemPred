# Benchmarks

The active independent evaluation is the pinned Banking77 experiment. Its
dataset fetcher verifies upstream hashes, the multi-intent TF-IDF baseline is
in `banking77.py`, and the frozen-embedding few-shot experiment is in
`banking77_fewshot.py`. See [`results/banking77-fewshot.json`](results/banking77-fewshot.json)
for the full 8- and 16-example, ten-seed report.

```bash
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

Older single-predicate NLI/LLM results and synthetic-corpus development runs
are historical only; they are not comparable to the 77-intent scores and do
not support product claims. No synthetic-data generator or generated-data
training default is included. Keep downloaded data, model weights, and
customer corpora out of Git.
