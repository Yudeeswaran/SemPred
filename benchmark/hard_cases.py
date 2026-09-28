from sklearn.model_selection import train_test_split

from sempred import SemPred
from training.synthetic import generate

HARD = [
    ("I need my money returned for this purchase.", "customer is requesting a refund", 1),
    ("The merchant already sent the money back to my card.", "customer is requesting a refund", 0),
    ("The courier showed up after the promised date.", "customer experienced a delivery problem", 1),
    ("Could you explain how to change where my package is delivered?", "customer experienced a delivery problem", 0),
    ("Please end the recurring membership immediately.", "customer wants to cancel a subscription", 1),
    ("I cancelled it two weeks ago, but it is still visible in my account.", "customer wants to cancel a subscription", 0),
    ("There is a second charge for the same purchase.", "customer has a billing problem", 1),
    ("My invoice total matches what I expected.", "customer has a billing problem", 0),
]


def main():
    rows = generate(500)
    train, _ = train_test_split(rows, test_size=.25, random_state=42, stratify=[r.family for r in rows])
    model = SemPred.fit([r.text for r in train], [r.predicate for r in train], [r.label for r in train])
    correct = 0
    for text, predicate, expected in HARD:
        result = model.predict(text, predicate)
        correct += int(result.label == bool(expected))
        print(f"expected={expected} predicted={int(result.label)} p={result.probability:.3f} :: {text}")
    print(f"hard_accuracy={correct / len(HARD):.4f}")


if __name__ == "__main__":
    main()
