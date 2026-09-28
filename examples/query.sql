-- Requires the optional DuckDB integration to be installed.
-- Intended V1 syntax:
SELECT *
FROM tickets
WHERE SEM_SCORE(ticket_text, 'customer is requesting a refund') > 0.90;
