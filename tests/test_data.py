from __future__ import annotations

import json
import unittest
from collections import Counter
from pathlib import Path

from vgcf.data import (
    DatasetSchemaError,
    assert_safe_model_input,
    balanced_sample,
    build_model_input,
    fixed_sample,
    inspect_proofwriter_dataset,
    load_examples,
)

from tests.helpers import example


class DataTests(unittest.TestCase):
    def test_nested_adapter_does_not_leak_proofs_or_logic_representations(self) -> None:
        sentinel = "DO_NOT_LEAK_SECRET_91E2"
        record = {
            "id": "story-1",
            "triples": {
                "triple1": {
                    "text": "Alice is red.",
                    "representation": sentinel,
                }
            },
            "rules": {},
            "questions": {
                "Q1": {
                    "question": "Alice is red.",
                    "answer": True,
                    "allProofs": sentinel,
                }
            },
        }
        path = (
            Path(__file__).resolve().parents[1]
            / "outputs"
            / "test_runtime"
            / "leakage_data.jsonl"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        loaded = load_examples(path)[0]
        payload = build_model_input(loaded)
        serialized = json.dumps(payload)
        self.assertEqual(set(payload), {"theory", "question"})
        self.assertNotIn(sentinel, serialized)
        self.assertNotIn("answer", serialized)
        self.assertNotIn("allProofs", serialized)
        self.assertNotIn("representation", serialized)

    def test_whitelist_rejects_extra_fields(self) -> None:
        with self.assertRaises(DatasetSchemaError):
            assert_safe_model_input(
                {"theory": ["Alice is red."], "question": "Alice is red.", "answer": True}
            )

    def test_fixed_sampling_is_reproducible(self) -> None:
        examples = [example([f"Alice is p{i}."], f"Alice is p{i}.") for i in range(10)]
        object.__setattr__(examples[0], "example_id", "first")
        for index, item in enumerate(examples):
            object.__setattr__(item, "example_id", f"e{index}")
        first = [item.example_id for item in fixed_sample(examples, 5, 42)]
        second = [item.example_id for item in fixed_sample(examples, 5, 42)]
        self.assertEqual(first, second)

    def test_balanced_sampling_is_reproducible_and_label_balanced(self) -> None:
        examples = []
        for label in ("True", "False", "Unknown"):
            for index in range(6):
                item = example(
                    [
                        "Alice is not red."
                        if index == 0
                        else "Alice visits Bob."
                    ],
                    "Alice is red." if index % 2 == 0 else "Alice visits Bob.",
                    label,
                )
                object.__setattr__(item, "example_id", f"{label}-{index}")
                object.__setattr__(item, "theory_id", f"story-{label}-{index}")
                object.__setattr__(item, "depth", index % 3)
                examples.append(item)
        first = balanced_sample(examples, 10, 20260805)
        second = balanced_sample(examples, 10, 20260805)
        self.assertEqual(
            [item.example_id for item in first],
            [item.example_id for item in second],
        )
        self.assertEqual(
            Counter(item.gold_label for item in first),
            Counter({"True": 3, "False": 3, "Unknown": 4}),
        )
        self.assertGreater(len({item.depth for item in first}), 1)

    def test_full_inspection_reports_annotations_without_exposing_proof_text(self) -> None:
        sentinel = "SECRET_PROOF_SENTINEL"
        record = {
            "id": "RelNeg-OWA-D1-test",
            "maxD": 1,
            "NFact": 1,
            "NRule": 0,
            "theory": "Alice is red.",
            "triples": {
                "triple1": {"text": "Alice is red.", "representation": sentinel}
            },
            "rules": {},
            "questions": {
                "Q1": {
                    "question": "Alice is red.",
                    "answer": True,
                    "QDep": 0,
                    "proofs": sentinel,
                    "representation": sentinel,
                }
            },
            "allProofs": sentinel,
            "proofDetails": [],
        }
        path = (
            Path(__file__).resolve().parents[1]
            / "outputs"
            / "test_runtime"
            / "inspection.jsonl"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        report = inspect_proofwriter_dataset(path)
        serialized = json.dumps(report)
        self.assertEqual(report["label_distribution"], {"True": 1})
        self.assertEqual(report["theory_text_mismatch_count"], 0)
        self.assertNotIn(sentinel, serialized)


if __name__ == "__main__":
    unittest.main()
