from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class Prediction:
    probability: float
    label: bool
    decision: Literal["true", "false", "unknown"] = "unknown"


class SemPred:
    """Legacy TF-IDF/logistic-regression baseline.

    This remains available for baseline comparisons and existing safe
    JSON/NumPy model archives. The frozen-embedding experiment is implemented
    by :class:`sempred.fewshot.FewShotSemPred`.
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
        self._safe_vectorizer: TfidfVectorizer | None = None
        self._safe_coefficients: np.ndarray | None = None
        self._safe_intercept: float | None = None

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
            probability = float(self._predict_proba([self._pair(text, predicate)])[0])
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
            scores = self._predict_proba(pairs)
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

    def _predict_proba(self, pairs: Sequence[str]) -> np.ndarray:
        if self._safe_vectorizer is None:
            return self.pipeline.predict_proba(pairs)[:, 1]
        features = self._safe_vectorizer.transform(pairs)
        logits = np.asarray(features @ self._safe_coefficients.T).reshape(-1) + self._safe_intercept
        return expit(logits)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._safe_vectorizer is None:
            vectorizer = self.pipeline.named_steps["tfidf"]
            classifier = self.pipeline.named_steps["classifier"]
            vocabulary = vectorizer.vocabulary_
            idf = vectorizer.idf_
            coefficients = classifier.coef_
            intercept = classifier.intercept_
        else:
            vectorizer = self._safe_vectorizer
            vocabulary = vectorizer.vocabulary_
            idf = vectorizer.idf_
            coefficients = self._safe_coefficients.reshape(1, -1)
            intercept = np.asarray([self._safe_intercept])

        vectorizer_params = vectorizer.get_params()
        config = {
            "format": "sempred-tfidf",
            "format_version": 1,
            "threshold": self.threshold,
            "abstain_margin": self.abstain_margin,
            "cache_size": self.cache_size,
            "vocabulary": {str(token): int(index) for token, index in vocabulary.items()},
            "vectorizer": {
                "analyzer": vectorizer_params["analyzer"],
                "ngram_range": list(vectorizer_params["ngram_range"]),
                "lowercase": vectorizer_params["lowercase"],
                "strip_accents": vectorizer_params["strip_accents"],
                "token_pattern": vectorizer_params["token_pattern"],
                "stop_words": vectorizer_params["stop_words"],
                "norm": vectorizer_params["norm"],
                "use_idf": vectorizer_params["use_idf"],
                "smooth_idf": vectorizer_params["smooth_idf"],
                "sublinear_tf": vectorizer_params["sublinear_tf"],
                "binary": vectorizer_params["binary"],
            },
        }
        arrays = io.BytesIO()
        np.savez_compressed(
            arrays,
            idf=np.asarray(idf, dtype=np.float64),
            coefficients=np.asarray(coefficients, dtype=np.float64),
            intercept=np.asarray(intercept, dtype=np.float64),
        )
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("model.json", json.dumps(config, separators=(",", ":")))
            archive.writestr("parameters.npz", arrays.getvalue())

    @classmethod
    def load(cls, path: str | Path) -> SemPred:
        path = Path(path)
        if not zipfile.is_zipfile(path):
            raise ValueError(
                "unsupported SemPred model format; pickle models are not loaded"
            )
        with zipfile.ZipFile(path) as archive:
            config = json.loads(archive.read("model.json"))
            if config.get("format") != "sempred-tfidf" or config.get("format_version") != 1:
                raise ValueError("unsupported SemPred model archive version")
            vocabulary = config.get("vocabulary")
            if not isinstance(vocabulary, dict) or not vocabulary:
                raise ValueError("model archive has an invalid vocabulary")
            vocabulary = {str(token): int(index) for token, index in vocabulary.items()}
            if sorted(vocabulary.values()) != list(range(len(vocabulary))):
                raise ValueError("model archive vocabulary indices are invalid")
            with np.load(io.BytesIO(archive.read("parameters.npz")), allow_pickle=False) as weights:
                idf = np.asarray(weights["idf"], dtype=np.float64)
                coefficients = np.asarray(weights["coefficients"], dtype=np.float64)
                intercept = np.asarray(weights["intercept"], dtype=np.float64)

        if idf.shape != (len(vocabulary),) or coefficients.shape != (1, len(vocabulary)) or intercept.shape != (1,):
            raise ValueError("model archive parameter dimensions do not match its vocabulary")
        if not (np.isfinite(idf).all() and np.isfinite(coefficients).all() and np.isfinite(intercept).all()):
            raise ValueError("model archive contains non-finite parameters")
        vectorizer_params = dict(config["vectorizer"])
        vectorizer_params["ngram_range"] = tuple(vectorizer_params["ngram_range"])
        vectorizer = TfidfVectorizer(**vectorizer_params)
        vectorizer.vocabulary_ = vocabulary
        vectorizer.fixed_vocabulary_ = True
        vectorizer.idf_ = idf

        model = cls.__new__(cls)
        model.pipeline = None
        model.threshold = float(config["threshold"])
        model.abstain_margin = float(config["abstain_margin"])
        model.cache_size = int(config["cache_size"])
        if not 0.0 <= model.threshold <= 1.0 or not 0.0 <= model.abstain_margin <= 1.0 or model.cache_size < 0:
            raise ValueError("model archive contains invalid runtime settings")
        model._cache = OrderedDict()
        model._safe_vectorizer = vectorizer
        model._safe_coefficients = coefficients[0]
        model._safe_intercept = float(intercept[0])
        return model
