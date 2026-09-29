"""Flag synthetic training rows whose wrappers may conflict with their labels.

The checks are review aids, not automatic relabeling rules. Predicate meaning
varies by family, so a human or independent annotation pass must resolve flags.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

REVIEW_RULES = {
    "hypothetical_marked_positive": lambda text, label: label == 1
    and ("only hypothetical" in text or "does not imply the target action" in text),
    "no_evidence_marked_positive": lambda text, label: label == 1
    and text.startswith((
        "There is no evidence that ",
        "It does not mean that ",
        "The note does not say that ",
        "Do not conclude that ",
    )),
    "other_speaker_marked_positive": lambda text, label: label == 1
    and text.startswith("Someone else said this: "),
    "historical_change_marked_positive": lambda text, label: label == 1
    and text.startswith("Earlier, ")
    and "current status is different" in text,
    "opposite_marked_negative": lambda text, label: label == 0
    and text.startswith("The customer explicitly says the opposite: "),
}


def audit(path: Path) -> dict:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not {"text", "label"}.issubset(row):
            raise ValueError(f"{path}:{line_number}: expected text and label fields")
        rows.append((line_number, row))

    findings = Counter()
    examples = []
    flagged_lines = set()
    for line_number, row in rows:
        text = row["text"]
        label = int(row["label"])
        for rule_name, rule in REVIEW_RULES.items():
            if rule(text, label):
                findings[rule_name] += 1
                flagged_lines.add(line_number)
                if sum(item["rule"] == rule_name for item in examples) < 5:
                    examples.append({
                        "line": line_number,
                        "rule": rule_name,
                        "label": label,
                        "family": row.get("family"),
                        "text": text,
                    })

    return {
        "dataset": str(path),
        "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": len(rows),
        "rows_flagged_for_review": len(flagged_lines),
        "review_counts": dict(sorted(findings.items())),
        "review_examples": examples,
        "warning": "These pattern flags are not confirmed mislabels; review them before training.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/sempred_32k_adversarial.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/training-data-audit.json"))
    args = parser.parse_args()
    report = audit(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "review_examples"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
