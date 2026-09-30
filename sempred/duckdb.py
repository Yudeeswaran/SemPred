from __future__ import annotations


def register(
    con,
    model,
    *,
    score_function: str = "SEM_SCORE",
    predicate_function: str = "SEM_PREDICT",
):
    """Register score and tri-state predicate functions on a DuckDB connection.

    ``SEM_PREDICT`` returns TRUE/FALSE for confident decisions and SQL NULL for
    SemPred's ``unknown`` decision. When PyArrow is installed, both functions
    use DuckDB's Arrow UDF path and score each chunk with ``predict_batch``.
    Without PyArrow they fall back to scalar Python UDFs.
    """
    try:
        import pyarrow as pa
    except ImportError:
        con.create_function(score_function, model.score, return_type="DOUBLE")

        def predicate(text: str | None, condition: str | None):
            if text is None or condition is None:
                return None
            result = model.predict(text, condition)
            return None if result.decision == "unknown" else result.label

        con.create_function(
            predicate_function,
            predicate,
            return_type="BOOLEAN",
            null_handling="special",
        )
    else:

        def arrow_udf(*, as_score: bool):
            return_type = pa.float64() if as_score else pa.bool_()

            def evaluate(texts, predicates):
                text_values = texts.to_pylist()
                predicate_values = predicates.to_pylist()
                output = [None] * len(text_values)
                valid = [
                    index
                    for index, (text, condition) in enumerate(
                        zip(text_values, predicate_values)
                    )
                    if text is not None and condition is not None
                ]
                if hasattr(model, "predict_many"):
                    predictions = model.predict_many(
                        [text_values[index] for index in valid],
                        [predicate_values[index] for index in valid],
                    )
                    groups = [(valid, predictions)]
                else:
                    grouped: dict[str, list[int]] = {}
                    for index in valid:
                        grouped.setdefault(predicate_values[index], []).append(index)
                    groups = [
                        (
                            indexes,
                            model.predict_batch(
                                [text_values[i] for i in indexes], condition
                            ),
                        )
                        for condition, indexes in grouped.items()
                    ]
                for indexes, predictions in groups:
                    for index, prediction in zip(indexes, predictions):
                        if as_score:
                            output[index] = prediction.probability
                        elif prediction.decision != "unknown":
                            output[index] = prediction.label
                return pa.array(output, type=return_type)

            return evaluate

        con.create_function(
            score_function,
            arrow_udf(as_score=True),
            return_type="DOUBLE",
            type="arrow",
            null_handling="special",
        )
        con.create_function(
            predicate_function,
            arrow_udf(as_score=False),
            return_type="BOOLEAN",
            type="arrow",
            null_handling="special",
        )
    return con
