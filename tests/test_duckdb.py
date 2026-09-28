import duckdb

from sempred import SemPred
from sempred.duckdb import register


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
