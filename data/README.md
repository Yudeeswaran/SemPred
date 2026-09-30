# Local data

Generated and customer-provided corpora are kept out of Git. The synthetic
generator in `training/synthetic.py` is for examples and software checks only;
its template-based examples are not independent evaluation data.

Use the hash-pinned public-data fetcher and benchmark under `benchmark/` for
the independent Banking77 evaluation. Historical results are retained in
`docs/EVALUATION.md` and `benchmark/results/`; they are not release claims.
