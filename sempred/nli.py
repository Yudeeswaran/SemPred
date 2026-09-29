"""Pretrained NLI cross-encoder backend for semantic predicate scoring.

The backend is optional and loads Hugging Face models as local files after its
first download. It uses entailment probability as a score, not as a calibrated
probability unless the caller has calibrated it on representative data.
"""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path

from .calibration import apply_binary_calibration, calibrate_binary_scores
from .core import Prediction, SemPred

DEFAULT_MODEL_ID = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
DEFAULT_MODEL_REVISION = "6f5cf0a2b59cabb106aca4c287eed12e357e90eb"
_CONFIG_NAME = "sempred-nli.json"


class NLISemPred:
    """Natural-language inference cross-encoder with a SemPred-compatible API."""

    def __init__(
        self,
        tokenizer,
        model,
        *,
        device: str = "cpu",
        threshold: float = 0.5,
        abstain_margin: float = 0.0,
        cache_size: int = 10_000,
        max_length: int = 256,
        batch_size: int = 32,
        model_id: str = DEFAULT_MODEL_ID,
        revision: str | None = DEFAULT_MODEL_REVISION,
        hypothesis_template: str = "It is true that {predicate}.",
    ):
        import torch

        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if not 0.0 <= abstain_margin <= 1.0:
            raise ValueError("abstain_margin must be between 0 and 1")
        if cache_size < 0:
            raise ValueError("cache_size must be zero or greater")
        if max_length < 8:
            raise ValueError("max_length must be at least 8")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if hypothesis_template.count("{predicate}") != 1:
            raise ValueError("hypothesis_template must contain exactly one {predicate} placeholder")
        if max_length > getattr(model.config, "max_position_embeddings", max_length):
            raise ValueError("max_length exceeds the model's positional-embedding limit")

        labels = {int(key): str(value).strip().lower() for key, value in model.config.id2label.items()}
        entailment = [index for index, label in labels.items() if label == "entailment"]
        if len(entailment) != 1:
            raise ValueError(f"model must expose exactly one entailment label; found {labels}")

        self.torch = torch
        self.tokenizer = tokenizer
        self.model = model
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA device")
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self.entailment_index = entailment[0]
        self.threshold = float(threshold)
        self.abstain_margin = float(abstain_margin)
        self.cache_size = int(cache_size)
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.model_id = model_id
        self.revision = revision
        self.hypothesis_template = hypothesis_template
        self.calibration_scale = 1.0
        self.calibration_bias = 0.0
        self.calibration_metadata: dict | None = None
        self._cache: OrderedDict[str, float] = OrderedDict()

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        revision: str | None = DEFAULT_MODEL_REVISION,
        device: str = "cpu",
        local_files_only: bool = False,
        cache_dir: str | Path | None = None,
        threshold: float = 0.5,
        abstain_margin: float = 0.0,
        cache_size: int = 10_000,
        max_length: int = 256,
        batch_size: int = 32,
        hypothesis_template: str = "It is true that {predicate}.",
    ) -> NLISemPred:
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install SemPred's NLI dependencies with `pip install 'sempred[nli]'`") from exc

        tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, local_files_only=local_files_only, cache_dir=cache_dir
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
            use_safetensors=True,
        )
        return cls(
            tokenizer,
            model,
            device=device,
            threshold=threshold,
            abstain_margin=abstain_margin,
            cache_size=cache_size,
            max_length=max_length,
            batch_size=batch_size,
            model_id=model_id,
            revision=revision,
            hypothesis_template=hypothesis_template,
        )

    def _hypothesis(self, predicate: str) -> str:
        return self.hypothesis_template.format(predicate=predicate.rstrip(" .?!"))

    @staticmethod
    def _key(text: str, predicate: str) -> str:
        return hashlib.sha256(f"{predicate}\0{text}".encode()).hexdigest()

    def _remember(self, key: str, probability: float) -> None:
        if not self.cache_size:
            return
        self._cache[key] = probability
        self._cache.move_to_end(key)
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def _infer(self, texts: list[str], predicate: str) -> list[float]:
        scores = []
        hypotheses = [self._hypothesis(predicate)] * len(texts)
        for offset in range(0, len(texts), self.batch_size):
            batch_texts = texts[offset : offset + self.batch_size]
            batch_hypotheses = hypotheses[offset : offset + self.batch_size]
            tokens = self.tokenizer(
                batch_texts,
                batch_hypotheses,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            tokens = {name: tensor.to(self.device) for name, tensor in tokens.items()}
            with self.torch.inference_mode():
                logits = self.model(**tokens).logits
                probs = self.torch.softmax(logits, dim=-1)[:, self.entailment_index]
            scores.extend(float(value) for value in probs.cpu().tolist())
        return scores

    def _calibrate_score(self, probability: float) -> float:
        return float(apply_binary_calibration([probability], self.calibration_scale, self.calibration_bias)[0])

    def predict(self, text: str, predicate: str) -> Prediction:
        SemPred._validate_input(text, predicate)
        key = self._key(text, predicate)
        if key in self._cache:
            self._cache.move_to_end(key)
            probability = self._cache[key]
        else:
            probability = self._calibrate_score(self._infer([text], predicate)[0])
            self._remember(key, probability)
        return self._prediction(probability)

    def score(self, text: str, predicate: str) -> float:
        return self.predict(text, predicate).probability

    def predict_batch(self, texts: Iterable[str], predicate: str) -> list[Prediction]:
        texts = list(texts)
        if not isinstance(predicate, str) or not predicate.strip():
            raise ValueError("predicate must be a non-empty string")
        for text in texts:
            SemPred._validate_input(text, predicate)

        keys = [self._key(text, predicate) for text in texts]
        probabilities: dict[str, float] = {}
        missing: dict[str, str] = {}
        for key, text in zip(keys, texts):
            if key in self._cache:
                self._cache.move_to_end(key)
                probabilities[key] = self._cache[key]
            else:
                missing[key] = text
        if missing:
            for key, probability in zip(missing, self._infer(list(missing.values()), predicate)):
                probability = self._calibrate_score(probability)
                probabilities[key] = probability
                self._remember(key, probability)
        return [self._prediction(probabilities[key]) for key in keys]

    def calibrate(self, texts: Iterable[str], predicates: Iterable[str], labels: Iterable[int | bool]) -> tuple[float, float]:
        """Fit Platt calibration on a separate representative held-out set."""
        texts, predicates, labels = list(texts), list(predicates), list(labels)
        if not texts or not (len(texts) == len(predicates) == len(labels)):
            raise ValueError("texts, predicates and labels must be non-empty and have equal lengths")
        if any(label not in (0, 1, False, True) for label in labels):
            raise ValueError("labels must be binary values 0 or 1")
        if len({int(label) for label in labels}) != 2:
            raise ValueError("calibration labels must contain both binary classes")
        for text, predicate in zip(texts, predicates):
            SemPred._validate_input(text, predicate)

        raw = [0.0] * len(texts)
        grouped: dict[str, list[int]] = {}
        for index, predicate in enumerate(predicates):
            grouped.setdefault(predicate, []).append(index)
        for predicate, indices in grouped.items():
            for start in range(0, len(indices), self.batch_size):
                selected = indices[start : start + self.batch_size]
                scores = self._infer([texts[index] for index in selected], predicate)
                for index, score in zip(selected, scores):
                    raw[index] = score

        self.calibration_scale, self.calibration_bias = calibrate_binary_scores(raw, labels)
        self._cache.clear()
        return self.calibration_scale, self.calibration_bias

    def _prediction(self, probability: float) -> Prediction:
        label = probability >= self.threshold
        decision = "unknown" if abs(probability - self.threshold) < self.abstain_margin else ("true" if label else "false")
        return Prediction(probability=probability, label=label, decision=decision)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path, safe_serialization=True)
        self.tokenizer.save_pretrained(path)
        config = {
            "backend": "nli-cross-encoder",
            "model_id": self.model_id,
            "revision": self.revision,
            "threshold": self.threshold,
            "abstain_margin": self.abstain_margin,
            "cache_size": self.cache_size,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "hypothesis_template": self.hypothesis_template,
            "calibration_scale": self.calibration_scale,
            "calibration_bias": self.calibration_bias,
            "calibration_metadata": self.calibration_metadata,
        }
        (path / _CONFIG_NAME).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, *, device: str = "cpu") -> NLISemPred:
        path = Path(path)
        try:
            config = json.loads((path / _CONFIG_NAME).read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ValueError(f"{path} is not a saved SemPred NLI model directory") from exc
        model = cls.from_pretrained(
            str(path),
            revision=None,
            device=device,
            local_files_only=True,
            threshold=config["threshold"],
            abstain_margin=config["abstain_margin"],
            cache_size=config["cache_size"],
            max_length=config["max_length"],
            batch_size=config["batch_size"],
            hypothesis_template=config.get("hypothesis_template", "{predicate}"),
        )
        model.model_id = config["model_id"]
        model.revision = config["revision"]
        model.calibration_scale = config.get("calibration_scale", 1.0)
        model.calibration_bias = config.get("calibration_bias", 0.0)
        model.calibration_metadata = config.get("calibration_metadata")
        return model
