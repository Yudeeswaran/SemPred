# Local data

Customer corpora and downloaded benchmark data are kept out of Git. SemPred
does not generate synthetic training examples. Training and evaluation should
use real labeled data with documented provenance and separate train/test splits.

Use the hash-pinned public-data fetcher and benchmark under `benchmark/` for
the independent Banking77 evaluation. Historical results are retained in
`docs/EVALUATION.md` and `benchmark/results/`; they are not release claims.
