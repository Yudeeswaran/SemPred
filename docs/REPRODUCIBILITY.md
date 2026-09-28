# Reproducibility

Every accepted model should record:

- source commit SHA;
- Python version;
- dependency lock/version information;
- random seed;
- dataset name and SHA-256;
- train/validation/test split definition;
- model identifier and model SHA when applicable;
- tokenizer/version;
- maximum sequence length;
- training hyperparameters;
- calibration parameters;
- benchmark command;
- raw predictions or a deterministic prediction artifact;
- final metrics.

## Rules

1. Synthetic data is useful for development but cannot by itself establish product quality.
2. A test set must not be regenerated from the same templates used for training.
3. Failed experiments remain documented.
4. Do not promote a model because of a single favorable metric.
5. Report coverage whenever abstention is used.
6. Report latency and throughput on fixed hardware/environment information.
7. External benchmark claims must cite the dataset/model source and license.

## Current limitation

The current development environment cannot download pretrained checkpoints or install missing packages from the public package index. Therefore the pretrained semantic-model gate is intentionally unresolved.
