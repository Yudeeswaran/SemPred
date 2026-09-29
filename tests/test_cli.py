from __future__ import annotations

from pathlib import Path

import pytest

from sempred.cli import _train, build_parser


def test_train_defaults_to_nli_backend(tmp_path: Path):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"example","predicate":"condition","label":1}\n', encoding="utf-8")

    args = build_parser().parse_args(["train", "--data", str(data), "--model", str(tmp_path / "model")])

    assert args.backend == "nli"


def test_train_nli_reports_missing_base_model(tmp_path: Path):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"example","predicate":"condition","label":1}\n', encoding="utf-8")
    args = build_parser().parse_args([
        "train", "--data", str(data), "--model", str(tmp_path / "out"),
        "--base-model", str(tmp_path / "missing"),
    ])

    with pytest.raises(ValueError, match="download-nli"):
        _train(args)


def test_train_rejects_malformed_jsonl(tmp_path: Path):
    data = tmp_path / "train.jsonl"
    data.write_text('{broken json}\n', encoding="utf-8")
    args = build_parser().parse_args(["train", "--data", str(data), "--model", str(tmp_path / "out")])

    with pytest.raises(ValueError, match="invalid JSON"):
        _train(args)
