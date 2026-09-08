from __future__ import annotations

import json
import unittest
from pathlib import Path

from vgcf.gold import gold_solver_sanity, load_gold_programs, semantic_metrics
from vgcf.schema import Formalization, Rule
from vgcf.solver import ForwardChainingSolver

from tests.helpers import atom, constant, variable


class GoldParserTests(unittest.TestCase):
    def setUp(self) -> None:
        root = Path(__file__).resolve().parents[1]
        self.path = root / "outputs" / "test_runtime" / "gold_fixture.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        records = []
        for index, name in enumerate(("alice", "bob", "cora"), start=1):
            records.append(
                {
                    "id": f"gold-story-{index}",
                    "triples": {
                        "triple1": {
                            "text": f"{name} is green.",
                            "representation": f'("{name}" "is" "green" "+")',
                        },
                        "triple2": {
                            "text": f"{name} is not kind.",
                            "representation": f'("{name}" "is" "kind" "-")',
                        },
                    },
                    "rules": {
                        "rule1": {
                            "text": "If something is green and not kind then it is nice.",
                            "representation": '((("something" "is" "green" "+") ("something" "is" "kind" "~")) -> ("something" "is" "nice" "+"))',
                        }
                    },
                    "questions": {
                        "Q1": {
                            "question": f"{name} is nice.",
                            "answer": True,
                            "representation": f'("{name}" "is" "nice" "+")',
                        }
                    },
                }
            )
        self.path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

    def test_official_program_runs_with_hard_negative_rule_body(self) -> None:
        program = load_gold_programs(self.path)["gold-story-1:Q1"]
        self.assertEqual(ForwardChainingSolver().solve(program.formalization).label, "True")

    def test_gold_solver_sanity_is_reproducible(self) -> None:
        first = gold_solver_sanity(self.path, 3, 99)
        second = gold_solver_sanity(self.path, 3, 99)
        self.assertEqual(first, second)
        self.assertEqual(first["reproduction_rate"], 1.0)
        self.assertTrue(first["passes_threshold"])

    def test_semantic_diagnostic_only_normalizes_case_and_alpha_variables(self) -> None:
        predicted = Formalization(
            (atom("red", constant("Alice"), source_id="s1"),),
            (
                Rule(
                    (atom("red", variable("Person")),),
                    atom("warm", variable("Person")),
                    "s2",
                ),
            ),
            atom("warm", constant("Alice"), source_id="q1"),
        )
        gold = Formalization(
            (atom("red", constant("alice"), source_id="s1"),),
            (
                Rule(
                    (atom("red", variable("X")),),
                    atom("warm", variable("X")),
                    "s2",
                ),
            ),
            atom("warm", constant("alice"), source_id="q1"),
        )
        metrics = semantic_metrics(predicted, gold)
        self.assertFalse(metrics["exact_match"])
        self.assertTrue(metrics["normalized_diagnostic_match"])
        self.assertEqual(metrics["exact_match_interpretation"], "lower_bound")

    def test_semantic_diagnostic_does_not_stem_or_merge_predicates(self) -> None:
        predicted = Formalization(
            (atom("visits", constant("alice"), source_id="s1"),),
            (),
            atom("visits", constant("alice"), source_id="q1"),
        )
        gold = Formalization(
            (atom("visit", constant("alice"), source_id="s1"),),
            (),
            atom("visit", constant("alice"), source_id="q1"),
        )
        self.assertFalse(semantic_metrics(predicted, gold)["normalized_diagnostic_match"])


if __name__ == "__main__":
    unittest.main()
