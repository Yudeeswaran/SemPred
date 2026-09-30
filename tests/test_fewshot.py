import numpy as np

from sempred.fewshot import FewShotSemPred


class TinyEncoder:
    model_name = "test-encoder"
    revision = "test"
    dimension = 2
    max_length = 128
    batch_size = 32
    sort_by_length = True
    texts_encoded = 0
    encoder_batches = 0

    def encode(self, texts):
        self.texts_encoded += len(texts)
        self.encoder_batches += 1
        return np.asarray(
            [[1.0, 0.0] if "refund" in text else [0.0, 1.0] for text in texts],
            dtype=np.float32,
        )


def make_model(**kwargs):
    encoder = TinyEncoder()
    model = FewShotSemPred.fit(
        ["refund requested", "refund pending", "charge posted", "charge confirmed"],
        ["refund"] * 4,
        [1, 1, 0, 0],
        encoder,
        **kwargs,
    )
    return model, encoder


def test_fewshot_predict_many_deduplicates_embeddings_and_abstains():
    model, encoder = make_model(abstain_margin=1.0)
    predictions = model.predict_many(
        ["refund requested", "charge posted", "refund requested"],
        ["refund", "refund", "refund"],
    )

    assert len(predictions) == 3
    assert (
        encoder.texts_encoded == 6
    )  # four support texts plus two distinct query texts
    assert all(prediction.decision == "unknown" for prediction in predictions)


def test_fewshot_archive_round_trip_uses_safe_numpy_format(tmp_path):
    model, encoder = make_model(abstain_margin=0.0)
    archive = tmp_path / "fewshot.zip"
    model.save(archive)

    restored = FewShotSemPred.load(archive, encoder=encoder)

    assert restored.predict("refund request", "refund").label is True
    assert restored.predict("charge posted", "refund").label is False
