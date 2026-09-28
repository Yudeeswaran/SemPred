"""Load either a saved SemPred NLI artifact or a pinned local HF checkpoint."""
from pathlib import Path

from sempred.nli import NLISemPred


def load_nli_model(path: Path, *, device: str = "cpu", revision: str | None = None) -> NLISemPred:
    if (path / "sempred-nli.json").is_file():
        return NLISemPred.load(path, device=device)
    return NLISemPred.from_pretrained(
        str(path), revision=revision, local_files_only=True, device=device
    )
