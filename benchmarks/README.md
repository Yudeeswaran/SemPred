# Benchmarks

This directory contains benchmark code and tracked result summaries. Frozen
datasets are under `data/`; exploratory NLI candidate results are under
`benchmarks/results/`. Downloaded model weights and temporary run artifacts
remain outside source control.

The benchmark hierarchy is:

1. ordinary held-out;
2. controlled hard;
3. ultra-stress/compositional;
4. external independent datasets;
5. systems/throughput;
6. 1M+ row end-to-end.

Run the pinned local NLI model against a JSONL dataset with:

```bash
python -m benchmark.run_nli --model models/sempred-nli --dataset data/sempred_v7_stress_12k.jsonl --output benchmarks/results/run.json
```

The stress data has already been inspected during candidate development, so
new results against it are regression checks rather than blind evaluation.

Do not promote a model using only the ordinary benchmark.
