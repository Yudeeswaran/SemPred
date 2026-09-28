from __future__ import annotations

import hashlib
import pickle
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class Prediction:
    probability: float
    label: bool
    decision: Literal["true", "false", "unknown"] = "unknown"


class SemPred:
    """V1 semantic-predicate classifier.

    This first implementation deliberately uses a transparent TF-IDF + logistic
    regression cross-input baseline. The public API is designed to remain stable
    when the model is replaced by the compact encoder in V2.
    """

    def __init__(self, pipeline: Pipeline, threshold: float = 0.5, abstain_margin: float = 0.0, cache_size: int = 10_000):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if not 0.0 <= abstain_margin <= 1.0:
            raise ValueError("abstain_margin must be between 0 and 1")
        if cache_size < 0:
            raise ValueError("cache_size must be zero or greater")
        self.pipeline = pipeline
        self.threshold = float(threshold)
        self.abstain_margin = float(abstain_margin)
        self.cache_size = int(cache_size)
        self._cache: OrderedDict[str, float] = OrderedDict()

    @staticmethod
    def _pair(text: str, predicate: str) -> str:
        return f"predicate: {predicate}\ntext: {text}"

    @classmethod
    def fit(
        cls,
        texts: Sequence[str],
        predicates: Sequence[str],
        labels: Sequence[int | bool],
        *,
        threshold: float = 0.5,
        abstain_margin: float = 0.0,
        cache_size: int = 10_000,
        max_features: int = 200_000,
    ) -> SemPred:
        if not (len(texts) == len(predicates) == len(labels)):
            raise ValueError("texts, predicates and labels must have equal length")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if not 0.0 <= abstain_margin <= 1.0:
            raise ValueError("abstain_margin must be between 0 and 1")
        if cache_size < 0:
            raise ValueError("cache_size must be zero or greater")
        if max_features < 1:
            raise ValueError("max_features must be positive")
        if not texts:
            raise ValueError("at least one training example is required")
        if any(not isinstance(value, str) or not value.strip() for values in (texts, predicates) for value in values):
            raise ValueError("texts and predicates must be non-empty strings")
        if any(value not in (0, 1, False, True) for value in labels):
            raise ValueError("labels must be binary values 0 or 1")
        if len({int(value) for value in labels}) != 2:
            raise ValueError("labels must contain both binary classes 0 and 1")
        pairs = [cls._pair(t, p) for t, p in zip(texts, predicates)]
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=max_features)),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])
        pipeline.fit(pairs, np.asarray(labels, dtype=int))
        return cls(pipeline, threshold=threshold, abstain_margin=abstain_margin, cache_size=cache_size)

    def predict(self, text: str, predicate: str) -> Prediction:
        self._validate_input(text, predicate)
        key = hashlib.sha256(f"{predicate}\0{text}".encode()).hexdigest()
        if key in self._cache:
            self._cache.move_to_end(key)
            probability = self._cache[key]
        else:
            probability = float(self.pipeline.predict_proba([self._pair(text, predicate)])[0, 1])
            self._remember(key, probability)
        return self._prediction(probability)

    def score(self, text: str, predicate: str) -> float:
        return self.predict(text, predicate).probability

    def predict_batch(self, texts: Iterable[str], predicate: str) -> list[Prediction]:
        texts = list(texts)
        if not isinstance(predicate, str) or not predicate.strip():
            raise ValueError("predicate must be a non-empty string")
        for text in texts:
            self._validate_input(text, predicate)
        keys = [hashlib.sha256(f"{predicate}\0{text}".encode()).hexdigest() for text in texts]
        probabilities: dict[str, float] = {}
        missing: dict[str, str] = {}
        for key, text in zip(keys, texts):
            if key in self._cache:
                self._cache.move_to_end(key)
                probabilities[key] = self._cache[key]
            else:
                missing[key] = self._pair(text, predicate)
        if missing:
            pairs = list(missing.values())
            scores = self.pipeline.predict_proba(pairs)[:, 1]
            for key, probability in zip(missing, scores):
                probabilities[key] = float(probability)
                self._remember(key, float(probability))
        return [self._prediction(probabilities[key]) for key in keys]

    def _remember(self, key: str, probability: float) -> None:
        if not self.cache_size:
            return
        self._cache[key] = probability
        self._cache.move_to_end(key)
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    @staticmethod
    def _validate_input(text: str, predicate: str) -> None:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        if not isinstance(predicate, str) or not predicate.strip():
            raise ValueError("predicate must be a non-empty string")

    def _prediction(self, probability: float) -> Prediction:
        label = probability >= self.threshold
        decision = "unknown" if abs(probability - self.threshold) < self.abstain_margin else ("true" if label else "false")
        return Prediction(probability=probability, label=label, decision=decision)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> SemPred:
        with Path(path).open("rb") as f:
            model = pickle.load(f)
        if not isinstance(model, cls):
            raise TypeError("model file does not contain a SemPred model")
        # Normalize models serialized by earlier SemPred versions.
        model.cache_size = getattr(model, "cache_size", 10_000)
        cached = getattr(model, "_cache", {})
        model._cache = OrderedDict(cached)
        while len(model._cache) > model.cache_size:
            model._cache.popitem(last=False)
        return model
