from __future__ import annotations

import json
import unittest

from vgcf.ir_v2 import V2IR, parse_v2_ir
from vgcf.metrics import aggregate_records
from vgcf.validator import StaticValidator

from tests.helpers import example


class V2IRTests(unittest.TestCase):
    def test_fence_whitespace_unary_and_binary_are_exact(self) -> None:
        raw = """  ```json
        {"facts":[{"id":"s1","atom":"+bright(alice)"},
        {"id":"s2","atom":"+visits(alice,bob)"}],"rules":[],
        "query":{"id":"q1","atom":"-visits(bob,alice)"}}
        ```  """
        ir = parse_v2_ir(raw)
        formalization = ir.to_formalization()
        self.assertEqual(len(formalization.facts[0].arguments), 1)
        self.assertEqual(len(formalization.facts[1].arguments), 2)
        self.assertEqual(
            tuple(term.name for term in formalization.query.arguments),
            ("bob", "alice"),
        )
        self.assertTrue(formalization.query.negated)

    def test_unary_is_not_padded_to_binary(self) -> None:
        source = example(["Alice likes Bob."], "Alice likes Bob.")
        raw = json.dumps(
            {
                "facts": [{"id": "s1", "atom": "+likes(alice)"}],
                "rules": [],
                "query": {"id": "q1", "atom": "+likes(alice,bob)"},
            }
        )
        result = StaticValidator().validate_json(raw, source)
        self.assertTrue(result.schema_valid)
        self.assertFalse(result.hard_valid)
        self.assertIn("arity_mismatch", {issue.code for issue in result.issues})
        assert result.formalization is not None
        self.assertEqual(len(result.formalization.facts[0].arguments), 1)

    def test_variable_query_fails_before_solver(self) -> None:
        source = example(["Alice is warm."], "Alice is warm.")
        raw = json.dumps(
            {
                "facts": [{"id": "s1", "atom": "+warm(alice)"}],
                "rules": [],
                "query": {"id": "q1", "atom": "+warm(X)"},
            }
        )
        result = StaticValidator().validate_json(raw, source)
        self.assertTrue(result.json_parse_valid)
        self.assertTrue(result.schema_valid)
        self.assertFalse(result.hard_valid)
        self.assertIn("non_ground_query", {issue.code for issue in result.issues})

    def test_four_operational_stages_are_not_collapsed(self) -> None:
        base = {
            "method": "constrained",
            "gold_label": "True",
            "predicted_label": "Error",
            "validation_errors": [],
            "repair_attempted": False,
            "repair_improved": False,
            "soft_issue_count": 0,
            "route_source": "method_error",
            "coverage": False,
            "fallback_used": False,
            "infrastructure_error": False,
            "method_error": True,
            "call_count": 1,
            "final_label_correct": False,
            "latency_ms": 0,
            "tokens": {"prompt": 0, "completion": 0, "total": 0},
            "cost_usd": 0,
            "cached": False,
            "depth": 0,
        }
        stages = [
            (False, False, False, False),
            (True, False, False, False),
            (True, True, False, True),
            (True, True, True, False),
        ]
        rows = [
            {
                **base,
                "json_parse_valid": json_ok,
                "schema_valid": schema_ok,
                "hard_validator_pass": hard_ok,
                "solver_executable": solver_ok,
            }
            for json_ok, schema_ok, hard_ok, solver_ok in stages
        ]
        metrics = aggregate_records(rows)["constrained"]
        self.assertEqual(metrics["json_parse_valid_count"], 3)
        self.assertEqual(metrics["schema_valid_count"], 2)
        self.assertEqual(metrics["hard_validator_pass_count"], 1)
        self.assertEqual(metrics["solver_executable_count"], 1)

    def test_schema_rejects_unknown_fields(self) -> None:
        with self.assertRaises(Exception):
            V2IR.from_dict({"facts": [], "rules": [], "query": {}, "answer": True})

    def test_ground_constant_case_normalization_is_deterministic(self) -> None:
        ir = parse_v2_ir(
            json.dumps(
                {
                    "facts": [
                        {"id": "s1", "atom": "+visits(Bob,The_Bald_Eagle)"},
                        {"id": "s2", "atom": "+visits(Charlie,The_Cat)"},
                    ],
                    "rules": [],
                    "query": {"id": "q1", "atom": "+visits(Bob,The_Bald_Eagle)"},
                }
            )
        ).to_formalization()
        self.assertEqual(
            tuple(term.name for term in ir.facts[0].arguments),
            ("bob", "the_bald_eagle"),
        )
        self.assertEqual(
            tuple(term.name for term in ir.facts[1].arguments),
            ("charlie", "the_cat"),
        )


if __name__ == "__main__":
    unittest.main()
