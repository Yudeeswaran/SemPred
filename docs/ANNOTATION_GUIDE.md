# Building a useful SemPred evaluation set

## Choose the workload first

An evaluation set only says something about the examples it represents. Start
with one customer-support workflow, such as deciding whether a message contains
a current refund request, cancellation intent, or billing dispute. Record the
intended input fields and the exact predicates a user will ask. Do not mix
unrelated security, reviews, support messages, and database events into one
headline accuracy score.

## Labeling rules

Each row needs the text, exact predicate, binary label, scenario group, source
type, and annotator identifier. A positive means the text supports the
predicate as written for the relevant actor and time. A negative means the
text does not support it; mere keyword overlap is not enough. Mark ambiguous,
underspecified, or internally conflicting examples for adjudication and keep
them out of the binary score until resolved.

Sample rows across ordinary cases and realistic failure modes: paraphrases,
negation, completed versus requested actions, past versus current state,
hypotheticals, quotations, another speaker's actions, multiple events, and
missing evidence. Vary wording and context naturally. Avoid producing many
examples by wrapping the same sentence in a small set of templates.

## Split and lock

Keep training examples separate from the final evaluation from collection
onward. Group related messages, paraphrases, and scenarios before splitting so
near-duplicates cannot cross partitions. Use a separate calibration set if the
product depends on probabilities or abstention. Resolve disagreements with a
second annotator or an adjudicator; report agreement and the unresolved count.

For the final set, save the exact JSONL, labeling instructions, source
description, split manifest, and SHA-256. Restrict access during model
development. Run each frozen candidate once at a predeclared threshold and
report accuracy, precision, recall, F1, calibration, and per-scenario results
with uncertainty intervals. Do not change prompts, thresholds, features, or
weights after seeing these results; any such change needs a fresh test set.

## Provenance

Record who supplied and labeled the examples and whether any tool assisted in
writing or adjudication. Assistant-generated examples can help test software
and expose obvious failure modes, but they are not independent human evidence
and must not be described as human-authored. Keep customer text private and
use approved data handling and consent practices for production examples.

## Data volume

Five hundred examples is a reasonable first diagnostic set, not proof of
production quality. Report confidence intervals and inspect category counts;
rare but costly errors require more targeted examples. Keep a separate,
representative training corpus and monitor reviewed production outcomes after
launch.
