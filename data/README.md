# Local research corpora

The JSONL files previously stored here were generated during early model
development. They use a small set of templated predicate families and are not
independent evaluation data. Keep bulky/generated corpora out of Git; place
local copies in this directory when reproducing historical training runs.

The historical scripts are named for what they do (for example,
`training/classical_baseline.py` and `training/compositional_stress.py`). Run
`python -m training.classical_baseline` to recreate the 24k base corpus and
`python -m training.adversarial_corpus` for the 32k corpus. The scripts run
their original training experiments as well as writing data. The 24k corpus
must exist before running `training.curriculum_training.py`; stress generation
also expects the baseline model artifact. Historical benchmark numbers are in
`docs/EVALUATION.md`, and exploratory run metadata is in `benchmark/results/`.
Do not use the development suites for final model selection or release claims.
