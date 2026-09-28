"""Fine-tune the pinned NLI encoder on SemPred-labeled JSONL data.

Validation families are held out as complete groups so identical predicates
do not leak across the train/validation split. Final stress data is never read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from sempred.nli import NLISemPred


def load_jsonl(paths: list[Path]) -> tuple[list[dict], dict[str, str]]:
    rows: list[dict] = []
    hashes = {}
    for path in paths:
        content = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(content).hexdigest()
        for line_no, line in enumerate(content.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not {"text", "predicate", "label", "family"}.issubset(row):
                raise ValueError(f"{path}:{line_no}: expected text, predicate, label, and family fields")
            if row["label"] not in (0, 1, False, True):
                raise ValueError(f"{path}:{line_no}: label must be binary")
            if not isinstance(row["text"], str) or not row["text"].strip() or not isinstance(row["predicate"], str) or not row["predicate"].strip():
                raise ValueError(f"{path}:{line_no}: text and predicate must be non-empty strings")
            rows.append(row)
    return rows, hashes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, default=Path("models/sempred-nli-v2"))
    parser.add_argument("--base-model-id", help="source model name for metadata when base-model is a raw local HF checkpoint")
    parser.add_argument("--base-revision", help="immutable source revision for metadata on raw HF checkpoints")
    parser.add_argument("--hypothesis-template", help="override hypothesis wording with one {predicate} placeholder")
    parser.add_argument("--train-data", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--validation-family-count", type=int, default=4)
    parser.add_argument(
        "--negative-nli-label",
        choices=("neutral", "contradiction"),
        default="neutral",
        help="NLI target for SemPred label 0; false does not always mean logical contradiction",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install the NLI and training extras before fine-tuning") from exc

    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("epochs, batch-size, and learning-rate must be positive")
    if not args.train_data:
        args.train_data = [Path("data/sempred_32k_adversarial.jsonl")]

    rows, dataset_hashes = load_jsonl(args.train_data)
    families = sorted({row["family"] for row in rows})
    if len(families) <= args.validation_family_count:
        raise ValueError("need more predicate families than validation-family-count")
    if len({int(row["label"]) for row in rows}) != 2:
        raise ValueError("training data must contain both binary classes")

    rng = random.Random(args.seed)
    shuffled_families = families.copy()
    rng.shuffle(shuffled_families)
    validation_families = set(shuffled_families[: args.validation_family_count])
    train_rows = [row for row in rows if row["family"] not in validation_families]
    validation_rows = [row for row in rows if row["family"] in validation_families]
    if len({int(row["label"]) for row in validation_rows}) != 2:
        raise ValueError("held-out validation families must contain both binary classes")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA device")
    device = torch.device(args.device)

    metadata_path = args.base_model / "sempred-nli.json"
    if metadata_path.is_file():
        base = NLISemPred.load(args.base_model, device=args.device)
        tokenizer = base.tokenizer
        model = base.model
        if args.hypothesis_template is not None:
            if args.hypothesis_template.count("{predicate}") != 1:
                parser.error("--hypothesis-template must contain exactly one {predicate} placeholder")
            base.hypothesis_template = args.hypothesis_template
    else:
        tokenizer = AutoTokenizer.from_pretrained(str(args.base_model), local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            str(args.base_model), local_files_only=True, use_safetensors=True
        ).to(device)
        hypothesis_template = args.hypothesis_template or "It is true that {predicate}."
        if hypothesis_template.count("{predicate}") != 1:
            parser.error("--hypothesis-template must contain exactly one {predicate} placeholder")
        base = NLISemPred(
            tokenizer,
            model,
            device=args.device,
            model_id=args.base_model_id or str(args.base_model),
            revision=args.base_revision,
            hypothesis_template=hypothesis_template,
        )
    nli_label_ids = {str(label).lower(): int(index) for index, label in model.config.id2label.items()}
    entailment_id = next((index for label, index in nli_label_ids.items() if label == "entailment"), None)
    negative_id = next((index for label, index in nli_label_ids.items() if label == args.negative_nli_label), None)
    if entailment_id is None or negative_id is None:
        raise ValueError(f"base model labels do not include entailment and {args.negative_nli_label}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    def make_loader(examples: list[dict], *, shuffle: bool) -> DataLoader:
        def collate(batch: list[dict]) -> dict:
            texts = [item["text"] for item in batch]
            hypotheses = [base._hypothesis(item["predicate"]) for item in batch]
            tokens = tokenizer(
                texts,
                hypotheses,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            tokens["labels"] = torch.tensor(
                [entailment_id if int(item["label"]) == 1 else negative_id for item in batch], dtype=torch.long
            )
            return tokens

        return DataLoader(examples, batch_size=args.batch_size, shuffle=shuffle, collate_fn=collate)

    train_loader = make_loader(train_rows, shuffle=True)
    validation_loader = make_loader(validation_rows, shuffle=False)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = max(1, int(total_steps * 0.06))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min((step + 1) / warmup_steps, max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))),
    )

    args.output.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    best_epoch = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader, 1):
            batch = {name: tensor.to(device) for name, tensor in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            output = model(**batch)
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += float(output.loss.detach().cpu())
            if step % 100 == 0 or step == len(train_loader):
                print(f"epoch={epoch} step={step}/{len(train_loader)} train_loss={total_loss / step:.4f}", flush=True)

        model.eval()
        validation_loss = 0.0
        validation_targets = []
        validation_probabilities = []
        count = 0
        with torch.inference_mode():
            for batch in validation_loader:
                batch = {name: tensor.to(device) for name, tensor in batch.items()}
                output = model(**batch)
                labels = batch["labels"]
                validation_loss += float(output.loss.detach().cpu()) * len(labels)
                validation_targets.extend(
                    int(row["label"]) for row in validation_rows[count : count + len(labels)]
                )
                validation_probabilities.extend(
                    float(value) for value in output.logits.softmax(dim=-1)[:, entailment_id].cpu().tolist()
                )
                count += len(labels)
        validation_loss /= count
        epoch_result = {
            "epoch": epoch,
            "train_loss": total_loss / len(train_loader),
            "validation_loss": validation_loss,
            "validation_accuracy": float(accuracy_score(validation_targets, np.asarray(validation_probabilities) >= 0.5)),
            "validation_f1": float(f1_score(validation_targets, np.asarray(validation_probabilities) >= 0.5)),
            "validation_roc_auc": float(roc_auc_score(validation_targets, validation_probabilities)),
        }
        history.append(epoch_result)
        print(json.dumps(epoch_result), flush=True)

        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            tuned = NLISemPred(
                tokenizer,
                model,
                device=args.device,
                threshold=base.threshold,
                abstain_margin=base.abstain_margin,
                cache_size=base.cache_size,
                max_length=args.max_length,
                batch_size=args.batch_size,
                model_id=base.model_id,
                revision=base.revision,
                hypothesis_template=base.hypothesis_template,
            )
            tuned.calibration_scale = 1.0
            tuned.calibration_bias = 0.0
            tuned.save(args.output)

    report = {
        "base_model": base.model_id,
        "base_revision": base.revision,
        "training_data_sha256": dataset_hashes,
        "training_examples": len(train_rows),
        "validation_examples": len(validation_rows),
        "validation_families": sorted(validation_families),
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "best_epoch": best_epoch,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "negative_nli_label": args.negative_nli_label,
        "nli_target_label_ids": {"negative": negative_id, "positive": entailment_id},
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
        "device": args.device,
    }
    (args.output / "training-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
