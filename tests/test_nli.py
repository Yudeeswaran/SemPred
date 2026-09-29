from __future__ import annotations

from typing import ClassVar

import pytest
import torch

from sempred.nli import NLISemPred


class FakeTokenizer:
    def __init__(self):
        self.calls: list[list[str]] = []

    def __call__(self, texts, hypotheses, **kwargs):
        self.calls.append(list(texts))
        size = len(texts)
        return {
            "input_ids": torch.ones((size, 4), dtype=torch.long),
            "attention_mask": torch.ones((size, 4), dtype=torch.long),
        }


class FakeModel:
    class Config:
        id2label: ClassVar[dict[int, str]] = {0: "contradiction", 1: "neutral", 2: "entailment"}
        max_position_embeddings = 32

    config = Config()

    def __init__(self):
        self.forward_calls = 0

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, **tokens):
        self.forward_calls += 1
        size = tokens["input_ids"].shape[0]
        return type("Output", (), {"logits": torch.tensor([[0.0, 0.0, 2.0]] * size)})()


def make_model(**kwargs):
    tokenizer = FakeTokenizer()
    model = FakeModel()
    return NLISemPred(tokenizer, model, batch_size=2, max_length=16, **kwargs), tokenizer, model


def test_predict_maps_entailment_and_reuses_cache():
    predictor, tokenizer, model = make_model()

    first = predictor.predict("The package is late.", "delivery was delayed")
    second = predictor.predict("The package is late.", "delivery was delayed")

    assert first.decision == "true"
    assert first.probability == pytest.approx(float(torch.softmax(torch.tensor([0.0, 0.0, 2.0]), 0)[2]))
    assert second == first
    assert len(tokenizer.calls) == 1
    assert model.forward_calls == 1


def test_batch_respects_batch_size_and_preserves_order():
    predictor, tokenizer, _ = make_model(cache_size=0)

    predictions = predictor.predict_batch(["first", "second", "third"], "condition is met")

    assert len(predictions) == 3
    assert [len(batch) for batch in tokenizer.calls] == [2, 1]


@pytest.mark.parametrize("text,predicate", [("", "valid"), ("valid", "  ")])
def test_predict_rejects_empty_inputs(text, predicate):
    predictor, _, _ = make_model()

    with pytest.raises(ValueError):
        predictor.predict(text, predicate)


def test_constructor_rejects_invalid_template():
    with pytest.raises(ValueError, match="placeholder"):
        make_model(hypothesis_template="This has no variable.")


def test_cache_is_bounded():
    predictor, _, _ = make_model(cache_size=1)
    predictor.predict("one", "condition")
    predictor.predict("two", "condition")

    assert len(predictor._cache) == 1
