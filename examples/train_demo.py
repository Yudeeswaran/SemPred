from pathlib import Path

from sklearn.model_selection import train_test_split

from benchmark.metrics import evaluate
from sempred import SemPred
from training.synthetic import generate


def main():
    rows = generate(n_per_family=500)
    train, test = train_test_split(rows, test_size=0.25, random_state=42, stratify=[r.family for r in rows])

    model = SemPred.fit(
        [r.text for r in train],
        [r.predicate for r in train],
        [r.label for r in train],
    )

    probs = [model.score(r.text, r.predicate) for r in test]
    metrics = evaluate([r.label for r in test], probs)
    print("SemPred V1 baseline")
    for k, v in metrics.items():
        print(f"{k:>18}: {v:.4f}")

    examples = [
        ("The package arrived three days late and I want my money back.", "customer is requesting a refund"),
        ("The package arrived exactly when promised.", "customer experienced a delivery problem"),
        ("Please stop charging me every month.", "customer wants to cancel a subscription"),
    ]
    print("\nPredictions")
    for text, predicate in examples:
        print(model.predict(text, predicate))

    out = Path("artifacts")
    out.mkdir(exist_ok=True)
    model.save(out / "sempred-v0.1.pkl")
    print(f"\nSaved: {out / 'sempred-v0.1.pkl'}")


if __name__ == "__main__":
    main()
