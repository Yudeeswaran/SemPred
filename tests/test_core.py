import zipfile

import pytest

from sempred import SemPred


def test_fit_predict():
    texts = [
        "I want my money back",
        "Please refund this payment",
        "The refund was already processed",
        "The product arrived on time",
    ]
    predicates = ["customer is requesting a refund"] * 4
    labels = [1, 1, 0, 0]
    model = SemPred.fit(texts, predicates, labels)
    assert 0.0 <= model.score("Please give me my money back", predicates[0]) <= 1.0


def test_batch():
    model = SemPred.fit(
        ["late package", "on time package"],
        ["delivery problem", "delivery problem"],
        [1, 0],
    )
    result = model.predict_batch(["late package", "on time package"], "delivery problem")
    assert len(result) == 2


def test_batch_handles_empty_input_and_bounded_cache():
    model = SemPred.fit(
        ["refund requested", "no refund requested"],
        ["customer requests a refund"] * 2,
        [1, 0],
        abstain_margin=1.0,
        cache_size=1,
    )

    assert model.predict_batch([], "customer requests a refund") == []
    predictions = model.predict_batch(
        ["refund requested", "no refund requested", "refund requested"],
        "customer requests a refund",
    )
    assert len(predictions) == 3
    assert all(item.decision == "unknown" for item in predictions)
    assert len(model._cache) <= 1


def test_fit_rejects_invalid_training_inputs():
    with pytest.raises(ValueError, match="both binary classes"):
        SemPred.fit(["a"], ["predicate"], [1])
    with pytest.raises(ValueError, match="non-empty strings"):
        SemPred.fit(["", "text"], ["predicate", "predicate"], [0, 1])


def test_safe_model_archive_round_trip(tmp_path):
    model = SemPred.fit(
        ["Please refund my payment", "refund was already processed", "the parcel is late", "the parcel arrived on time"],
        ["customer requests a refund", "customer requests a refund", "delivery is late", "delivery is late"],
        [1, 0, 1, 0],
    )
    path = tmp_path / "candidate.sempred"
    expected = model.predict_batch(
        ["I want my money back", "delivery came on time"], "customer requests a refund"
    )

    model.save(path)
    restored = SemPred.load(path)
    actual = restored.predict_batch(
        ["I want my money back", "delivery came on time"], "customer requests a refund"
    )

    assert zipfile.is_zipfile(path)
    assert [row.probability for row in actual] == pytest.approx(
        [row.probability for row in expected]
    )
    assert [row.decision for row in actual] == [row.decision for row in expected]


def test_loader_rejects_legacy_pickle_without_deserializing(tmp_path):
    path = tmp_path / "legacy.pkl"
    path.write_bytes(b"\x80\x04not-a-sempred-archive")

    with pytest.raises(ValueError, match="pickle models are not loaded"):
        SemPred.load(path)
