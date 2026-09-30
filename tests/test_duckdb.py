import duckdb

from sempred import SemPred
from sempred.duckdb import register
from sempred.fewshot import FewShotSemPred


class TinyEncoder:
    model_name = "test-encoder"
    revision = "test"
    dimension = 2
    texts_encoded = 0
    encoder_batches = 0

    def encode(self, texts):
        self.texts_encoded += len(texts)
        return [[1.0, 0.0] if "refund" in text else [0.0, 1.0] for text in texts]


def test_duckdb_functions_return_score_and_sql_null_for_unknown():
    model = SemPred.fit(
        ["refund requested", "no refund requested"],
        ["customer requests a refund"] * 2,
        [1, 0],
        abstain_margin=1.0,
    )
    con = register(duckdb.connect(), model)
    score, decision, null_input = con.execute(
        "SELECT SEM_SCORE('refund requested', 'customer requests a refund'), "
        "SEM_PREDICT('refund requested', 'customer requests a refund'), "
        "SEM_PREDICT(NULL, 'customer requests a refund')"
    ).fetchone()

    assert 0.0 <= score <= 1.0
    assert decision is None
    assert null_input is None
    con.close()


def test_duckdb_fewshot_arrow_udf_keeps_null_and_abstention():
    encoder = TinyEncoder()
    model = FewShotSemPred.fit(
        ["refund requested", "refund pending", "charge posted", "charge confirmed"],
        ["refund"] * 4,
        [1, 1, 0, 0],
        encoder,
        abstain_margin=1.0,
    )
    con = register(duckdb.connect(), model)

    rows = con.execute("""
        SELECT SEM_SCORE(text, predicate), SEM_PREDICT(text, predicate)
        FROM (VALUES ('refund requested', 'refund'), (NULL, 'refund')) AS input(text, predicate)
    """).fetchall()

    assert 0.0 <= rows[0][0] <= 1.0
    assert rows[0][1] is None
    assert rows[1] == (None, None)
    assert encoder.texts_encoded == 5  # four support texts plus one distinct query text
    con.close()
