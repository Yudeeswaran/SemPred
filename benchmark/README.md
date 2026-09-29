# Benchmark tools and results

This directory contains benchmark runners, metrics, and historical result
summaries. The files in `results/` are exploratory development evidence: the
underlying datasets were inspected during model development and are not blind
release evidence. Model weights and temporary run artifacts stay outside Git.

The benchmark hierarchy is:

1. ordinary held-out;
2. controlled hard;
3. ultra-stress/compositional;
4. external independent datasets;
5. systems/throughput;
6. 1M+ row end-to-end.

Run the pinned local NLI model against a JSONL dataset with:

```bash
python -m benchmark.run_nli --model models/sempred-nli --dataset data/sempred_v7_stress_12k.jsonl --output artifacts/run.json
```

The frozen-encoder baseline uses the pinned `all-MiniLM-L6-v2` sentence
encoder and a fixed logistic-regression classifier. Supply one or more
training corpora and a separate evaluation file:

```bash
python -m benchmark.embeddings_logreg --train-data data/sempred_32k_adversarial.jsonl --dataset locked/evaluation.jsonl --output artifacts/embeddings-baseline.json
```

The baseline encodes the text and predicate separately, then gives logistic
regression the two vectors, their absolute difference, and their elementwise
product. It does not tune the classifier or threshold on the evaluation set.

The old stress data has been inspected during candidate development, so new
results against it are regression checks rather than blind evaluation. The
large generated corpora are excluded from Git. Recreate the historical 24k
and 32k corpora with `python -m training.classical_baseline` and
`python -m training.adversarial_corpus`; these scripts also run their original
model experiments. Use a newly locked evaluation file for any release claim.

Do not promote a model using only the ordinary benchmark.
