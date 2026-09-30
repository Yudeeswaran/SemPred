"""Frozen sentence embeddings with one small binary head per predicate."""

from __future__ import annotations

import io
import json
import zipfile
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from .core import Prediction, SemPred

DEFAULT_ENCODER_ID = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_ENCODER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


class FrozenTextEncoder:
    """Mean-pool and normalize a pinned, frozen sentence encoder."""

    def __init__(
        self,
        model_name_or_path: str | Path = DEFAULT_ENCODER_ID,
        *,
        revision: str | None = DEFAULT_ENCODER_REVISION,
        device: str = "cpu",
        local_files_only: bool = True,
        max_length: int = 128,
        batch_size: int = 128,
    ):
        if max_length < 8 or batch_size < 1:
            raise ValueError(
                "max_length must be at least 8 and batch_size must be positive"
            )
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Install frozen-embedding dependencies with `pip install 'sempred[embeddings]'`"
            ) from exc

        source = str(model_name_or_path)
        local_path = Path(source)
        is_local = local_path.is_dir()
        metadata_path = local_path / "sempred-encoder.json" if is_local else None
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path and metadata_path.is_file()
            else {}
        )
        self.model_name = (
            metadata.get("model_name", DEFAULT_ENCODER_ID) if is_local else source
        )
        self.revision = metadata.get("revision") if is_local else revision
        self.device = torch.device(device)
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.texts_encoded = 0
        self.encoder_batches = 0
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            source,
            revision=None if is_local else revision,
            local_files_only=local_files_only or is_local,
        )
        self.model = AutoModel.from_pretrained(
            source,
            revision=None if is_local else revision,
            local_files_only=local_files_only or is_local,
            use_safetensors=True,
        ).to(self.device)
        self.model.eval()

    @classmethod
    def download(
        cls,
        output_dir: str | Path,
        *,
        model_name: str = DEFAULT_ENCODER_ID,
        revision: str = DEFAULT_ENCODER_REVISION,
        cache_dir: str | Path | None = None,
    ) -> Path:
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Install frozen-embedding dependencies with `pip install 'sempred[embeddings]'`"
            ) from exc
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=revision, cache_dir=cache_dir
        )
        model = AutoModel.from_pretrained(
            model_name, revision=revision, cache_dir=cache_dir, use_safetensors=True
        )
        tokenizer.save_pretrained(output_dir)
        model.save_pretrained(output_dir, safe_serialization=True)
        (output_dir / "sempred-encoder.json").write_text(
            json.dumps({"model_name": model_name, "revision": revision}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        return output_dir

    @property
    def dimension(self) -> int:
        return int(self.model.config.hidden_size)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        for index, text in enumerate(texts):
            SemPred._validate_input(text, "sentence")
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        self.texts_encoded += len(texts)
        self.encoder_batches += (len(texts) + self.batch_size - 1) // self.batch_size
        vectors = []
        with self.torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                tokens = self.tokenizer(
                    texts[start : start + self.batch_size],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                hidden = self.model(**tokens).last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                normalized = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
                vectors.append(normalized.cpu().numpy().astype(np.float32, copy=False))
        return np.concatenate(vectors, axis=0)


class FewShotSemPred:
    """A frozen encoder and logistic-regression head for each predicate."""

    FORMAT = "sempred-frozen-embeddings"
    FORMAT_VERSION = 1

    def __init__(
        self,
        encoder: FrozenTextEncoder,
        heads: dict[str, tuple[np.ndarray, float]],
        *,
        threshold: float = 0.5,
        abstain_margin: float = 0.1,
        cache_size: int = 10_000,
    ):
        if not 0.0 <= threshold <= 1.0 or not 0.0 <= abstain_margin <= 1.0:
            raise ValueError("threshold and abstain_margin must be between 0 and 1")
        if cache_size < 0:
            raise ValueError("cache_size must be zero or greater")
        if not heads:
            raise ValueError("at least one trained predicate head is required")
        self.encoder = encoder
        self.heads = {}
        for predicate, (coefficients, intercept) in heads.items():
            SemPred._validate_input(predicate, "predicate")
            vector = np.asarray(coefficients, dtype=np.float32).reshape(-1)
            if (
                vector.shape != (encoder.dimension,)
                or not np.isfinite(vector).all()
                or not np.isfinite(intercept)
            ):
                raise ValueError(f"invalid head parameters for predicate {predicate!r}")
            self.heads[predicate] = (vector, float(intercept))
        self.threshold = float(threshold)
        self.abstain_margin = float(abstain_margin)
        self.cache_size = int(cache_size)
        self._embedding_cache: OrderedDict[str, np.ndarray] = OrderedDict()

    @classmethod
    def fit(
        cls,
        texts: Sequence[str],
        predicates: Sequence[str],
        labels: Sequence[int | bool],
        encoder: FrozenTextEncoder,
        *,
        threshold: float = 0.5,
        abstain_margin: float = 0.1,
        cache_size: int = 10_000,
        c: float = 1.0,
    ) -> FewShotSemPred:
        texts, predicates, labels = list(texts), list(predicates), list(labels)
        if not texts or not (len(texts) == len(predicates) == len(labels)):
            raise ValueError(
                "texts, predicates and labels must be non-empty and have equal length"
            )
        if any(label not in (0, 1, False, True) for label in labels):
            raise ValueError("labels must be binary values 0 or 1")
        for text, predicate in zip(texts, predicates):
            SemPred._validate_input(text, predicate)
        embeddings = encoder.encode(texts)
        return cls.fit_embeddings(
            embeddings,
            predicates,
            labels,
            encoder,
            threshold=threshold,
            abstain_margin=abstain_margin,
            cache_size=cache_size,
            c=c,
        )

    @classmethod
    def fit_embeddings(
        cls,
        embeddings: np.ndarray,
        predicates: Sequence[str],
        labels: Sequence[int | bool],
        encoder: FrozenTextEncoder,
        *,
        threshold: float = 0.5,
        abstain_margin: float = 0.1,
        cache_size: int = 10_000,
        c: float = 1.0,
    ) -> FewShotSemPred:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        predicates, labels = list(predicates), list(labels)
        if embeddings.ndim != 2 or embeddings.shape != (
            len(predicates),
            encoder.dimension,
        ):
            raise ValueError(
                "embedding dimensions do not match predicate labels or encoder"
            )
        if len(predicates) != len(labels) or not predicates:
            raise ValueError(
                "predicates and labels must be non-empty and have equal length"
            )
        if any(label not in (0, 1, False, True) for label in labels):
            raise ValueError("labels must be binary values 0 or 1")
        if c <= 0 or not np.isfinite(embeddings).all():
            raise ValueError("c must be positive and embeddings must be finite")

        heads: dict[str, tuple[np.ndarray, float]] = {}
        groups: dict[str, list[int]] = {}
        for index, predicate in enumerate(predicates):
            SemPred._validate_input(predicate, "predicate")
            groups.setdefault(predicate, []).append(index)
        for predicate, indices in groups.items():
            target = np.asarray([int(labels[index]) for index in indices])
            if len(np.unique(target)) != 2:
                raise ValueError(
                    f"predicate {predicate!r} needs labeled positive and negative examples"
                )
            head = LogisticRegression(
                C=c, class_weight="balanced", max_iter=1000, solver="liblinear"
            )
            head.fit(embeddings[indices], target)
            heads[predicate] = (
                head.coef_[0].astype(np.float32),
                float(head.intercept_[0]),
            )
        return cls(
            encoder,
            heads,
            threshold=threshold,
            abstain_margin=abstain_margin,
            cache_size=cache_size,
        )

    def _embeddings_for(self, texts: list[str]) -> np.ndarray:
        for text in texts:
            SemPred._validate_input(text, "sentence")
        missing: dict[str, None] = {}
        for text in texts:
            if text in self._embedding_cache:
                self._embedding_cache.move_to_end(text)
            else:
                missing.setdefault(text, None)
        if missing:
            new_texts = list(missing)
            vectors = self.encoder.encode(new_texts)
            by_text = dict(zip(new_texts, vectors))
            if self.cache_size:
                for text, vector in zip(new_texts, vectors):
                    self._embedding_cache[text] = vector
                    self._embedding_cache.move_to_end(text)
                while len(self._embedding_cache) > self.cache_size:
                    self._embedding_cache.popitem(last=False)
            return np.stack(
                [
                    self._embedding_cache[text]
                    if text in self._embedding_cache
                    else by_text[text]
                    for text in texts
                ]
            )
        return np.stack([self._embedding_cache[text] for text in texts])

    def predict_many(
        self, texts: Sequence[str], predicates: Sequence[str]
    ) -> list[Prediction]:
        texts, predicates = list(texts), list(predicates)
        if len(texts) != len(predicates):
            raise ValueError("texts and predicates must have equal length")
        if not texts:
            return []
        for predicate in predicates:
            if predicate not in self.heads:
                raise ValueError(f"no trained head for predicate {predicate!r}")
        embeddings = self._embeddings_for(texts)
        logits = np.empty(len(texts), dtype=np.float32)
        groups: dict[str, list[int]] = {}
        for index, predicate in enumerate(predicates):
            groups.setdefault(predicate, []).append(index)
        for predicate, indices in groups.items():
            coefficients, intercept = self.heads[predicate]
            logits[indices] = embeddings[indices] @ coefficients + intercept
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
        return [self._prediction(float(probability)) for probability in probabilities]

    def predict_batch(self, texts: Sequence[str], predicate: str) -> list[Prediction]:
        if predicate not in self.heads:
            raise ValueError(f"no trained head for predicate {predicate!r}")
        return self.predict_many(texts, [predicate] * len(texts))

    def predict(self, text: str, predicate: str) -> Prediction:
        return self.predict_many([text], [predicate])[0]

    def score(self, text: str, predicate: str) -> float:
        return self.predict(text, predicate).probability

    def _prediction(self, probability: float) -> Prediction:
        label = probability >= self.threshold
        decision = (
            "unknown"
            if abs(probability - self.threshold) < self.abstain_margin
            else ("true" if label else "false")
        )
        return Prediction(probability=probability, label=label, decision=decision)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        predicates = list(self.heads)
        coefficients = np.stack([self.heads[predicate][0] for predicate in predicates])
        intercepts = np.asarray(
            [self.heads[predicate][1] for predicate in predicates], dtype=np.float32
        )
        config = {
            "format": self.FORMAT,
            "format_version": self.FORMAT_VERSION,
            "encoder": {
                "model_name": self.encoder.model_name,
                "revision": self.encoder.revision,
            },
            "threshold": self.threshold,
            "abstain_margin": self.abstain_margin,
            "cache_size": self.cache_size,
            "predicates": predicates,
        }
        arrays = io.BytesIO()
        np.savez_compressed(arrays, coefficients=coefficients, intercepts=intercepts)
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("model.json", json.dumps(config, separators=(",", ":")))
            archive.writestr("heads.npz", arrays.getvalue())

    @classmethod
    def load(
        cls, path: str | Path, *, encoder: FrozenTextEncoder | None = None
    ) -> FewShotSemPred:
        path = Path(path)
        if not zipfile.is_zipfile(path):
            raise ValueError("unsupported few-shot model archive")
        with zipfile.ZipFile(path) as archive:
            config = json.loads(archive.read("model.json"))
            if (
                config.get("format") != cls.FORMAT
                or config.get("format_version") != cls.FORMAT_VERSION
            ):
                raise ValueError("unsupported few-shot model archive version")
            with np.load(
                io.BytesIO(archive.read("heads.npz")), allow_pickle=False
            ) as arrays:
                coefficients = np.asarray(arrays["coefficients"], dtype=np.float32)
                intercepts = np.asarray(arrays["intercepts"], dtype=np.float32)
        if coefficients.ndim != 2 or intercepts.shape != (len(coefficients),):
            raise ValueError("few-shot archive has invalid head dimensions")
        if len(config["predicates"]) != len(coefficients) or not (
            np.isfinite(coefficients).all() and np.isfinite(intercepts).all()
        ):
            raise ValueError("few-shot archive has invalid head parameters")
        if encoder is None:
            encoder_config = config["encoder"]
            encoder = FrozenTextEncoder(
                encoder_config["model_name"],
                revision=encoder_config.get("revision"),
                local_files_only=True,
            )
        heads = {
            predicate: (coefficients[index], float(intercepts[index]))
            for index, predicate in enumerate(config["predicates"])
        }
        return cls(
            encoder,
            heads,
            threshold=config["threshold"],
            abstain_margin=config["abstain_margin"],
            cache_size=config["cache_size"],
        )

    def cache_info(self) -> dict[str, int]:
        return {
            "embeddings": len(self._embedding_cache),
            "max_embeddings": self.cache_size,
            "texts_encoded": self.encoder.texts_encoded,
            "encoder_batches": self.encoder.encoder_batches,
        }
