from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Example:
    text: str
    predicate: str
    label: int
    family: str


TEMPLATES = {
    "refund": [
        ("I want my money back for this order.", 1),
        ("Please refund the amount charged to me.", 1),
        ("Can you return my payment?", 1),
        ("The refund was processed yesterday.", 0),
        ("Can you tell me how long refunds take?", 0),
        ("I would like to exchange the item instead.", 0),
    ],
    "delivery_problem": [
        ("The package arrived three days late.", 1),
        ("My order has not arrived yet.", 1),
        ("The courier delivered the box damaged.", 1),
        ("The product arrived exactly on time.", 0),
        ("I like the product and its packaging.", 0),
        ("I need help changing my delivery address.", 0),
    ],
    "cancellation": [
        ("I want to cancel my subscription.", 1),
        ("Please stop my recurring plan.", 1),
        ("How do I cancel the service?", 1),
        ("I cancelled the subscription last month.", 0),
        ("I want to pause my subscription for a week.", 0),
        ("The subscription renewed successfully.", 0),
    ],
    "billing": [
        ("I was charged twice for the same order.", 1),
        ("There is an incorrect charge on my invoice.", 1),
        ("Why did my bill increase this month?", 1),
        ("My invoice looks correct.", 0),
        ("The product arrived late.", 0),
        ("I want to cancel my subscription.", 0),
    ],
}

PREDICATES = {
    "refund": "customer is requesting a refund",
    "delivery_problem": "customer experienced a delivery problem",
    "cancellation": "customer wants to cancel a subscription",
    "billing": "customer has a billing problem",
}


def generate(n_per_family: int = 250, seed: int = 7) -> list[Example]:
    rng = random.Random(seed)
    rows: list[Example] = []
    for family, items in TEMPLATES.items():
        for _ in range(n_per_family):
            text, label = rng.choice(items)
            rows.append(Example(text, PREDICATES[family], label, family))
    rng.shuffle(rows)
    return rows
