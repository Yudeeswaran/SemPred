"""Optional compact encoder backend.

Requires the `ml` extra. This module deliberately does not download a model;
callers provide a local Hugging Face-compatible encoder path.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError as exc:  # pragma: no cover
    AutoModel = AutoTokenizer = None
    _IMPORT_ERROR = exc


@dataclass
class EncoderConfig:
    model_name_or_path: str
    max_length: int = 256


class SemanticEncoder:
    def __init__(self, config: EncoderConfig):
        if AutoModel is None:
            raise RuntimeError("Install sempred[ml] to use the encoder backend") from _IMPORT_ERROR
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, local_files_only=True)
        self.encoder = AutoModel.from_pretrained(config.model_name_or_path, local_files_only=True)
        self.encoder.eval()

    @torch.inference_mode()
    def embed_pairs(self, texts: Sequence[str], predicates: Sequence[str]) -> torch.Tensor:
        batch = self.tokenizer(
            list(predicates),
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        output = self.encoder(**batch).last_hidden_state[:, 0]
        return torch.nn.functional.normalize(output, dim=-1)
