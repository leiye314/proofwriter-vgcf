from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any, Mapping

from vgcf.data import Example, SourceSentence
from vgcf.errors import SchemaError
from vgcf.ir_v2 import (
    normalize_singleton_rule_if,
    parse_v2_ir,
    parse_v2_ir_strict,
)
from vgcf.methods import MethodRunner
from vgcf.model import BaseModelClient, ClientConfig
from vgcf.parsing import (
    parse_cot_label,
    parse_cot_refine_label,
    parse_cot_refine_response,
    parse_cot_response,
    parse_label,
)
from vgcf.validator import StaticValidator


class BadRefineClient(BaseModelClient):
    def _request(
        self, payload: Mapping[str, Any]
    ) -> tuple[str, Mapping[str, Any], Mapping[str, Any]]:
        system = "\n".join(
            item["content"]
            for item in payload["messages"]
            if item["role"] == "system"
        )
        if "MODE=cot_refine" in system:
            content = (
                "REVIEW:\nThe original label is False.\n\n"
                "FINAL_LABEL_FOR_ORIGINAL_QUESTION: True\ntrailing"
            )
        else:
            content = (
                "REASONING:\nForward check complete.\n\n"
                "FINAL_LABEL_FOR_ORIGINAL_QUESTION: Unknown"
            )
        return (
            content,
            {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            {"model": "strict-refine-test"},
        )


class BadCotFallbackClient(BaseModelClient):
    def _request(
        self, payload: Mapping[str, Any]
    ) -> tuple[str, Mapping[str, Any], Mapping[str, Any]]:
        system = "\n".join(
            item["content"]
            for item in payload["messages"]
            if item["role"] == "system"
        )
        content = "Rationale says Unknown, then isolated True" if "MODE=cot" in system else "{}"
        return (
            content,
            {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            {"model": "bad-cot-fallback-test"},
        )


class CotLabelContractTests(unittest.TestCase):
    @staticmethod
    def _response(section: str, label: str) -> str:
        return (
            f"{section}:\nObject-level conclusion for the original question.\n\n"
            f"FINAL_LABEL_FOR_ORIGINAL_QUESTION: {label}"
        )

    def test_accepts_all_labels_and_exposes_visible_rationale(self) -> None:
        for label in ("True", "False", "Unknown"):
            with self.subTest(label=label):
                cot = parse_cot_response(self._response("REASONING", label))
                refine = parse_cot_refine_response(self._response("REVIEW", label))
                self.assertEqual(cot.label, label)
                self.assertEqual(refine.label, label)
                self.assertIn("original question", cot.rationale)
                self.assertIn("original question", refine.rationale)
                self.assertEqual(parse_cot_label(self._response("REASONING", label)), label)
                self.assertEqual(
                    parse_cot_refine_label(self._response("REVIEW", label)), label
                )

    def test_rejects_missing_duplicate_and_illegal_anchor_labels(self) -> None:
        invalid = (
            "REASONING:\nObject conclusion is True.",
            (
                "REASONING:\nFirst.\nFINAL_LABEL_FOR_ORIGINAL_QUESTION: True\n"
                "FINAL_LABEL_FOR_ORIGINAL_QUESTION: False"
            ),
            "REASONING:\nObject.\nFINAL_LABEL_FOR_ORIGINAL_QUESTION: true",
            "REASONING:\nObject.\nFINAL_LABEL_FOR_ORIGINAL_QUESTION: Maybe",
            "REASONING:\nObject.\nfinal_label_for_original_question: True",
        )
        for response in invalid:
            with self.subTest(response=response), self.assertRaises(SchemaError):
                parse_cot_label(response)

    def test_refine_has_the_same_missing_duplicate_illegal_and_trailing_boundaries(self) -> None:
        invalid = (
            "REVIEW:\nObject conclusion is Unknown.",
            (
                "REVIEW:\nObject.\nFINAL_LABEL_FOR_ORIGINAL_QUESTION: Unknown\n"
                "FINAL_LABEL_FOR_ORIGINAL_QUESTION: True"
            ),
            "REVIEW:\nObject.\nFINAL_LABEL_FOR_ORIGINAL_QUESTION: unknown",
            "REVIEW:\nObject.\nFINAL_LABEL_FOR_ORIGINAL_QUESTION: Invalid",
            (
                "REVIEW:\nObject.\nFINAL_LABEL_FOR_ORIGINAL_QUESTION: Unknown\n"
                "postscript"
            ),
        )
        for response in invalid:
            with self.subTest(response=response), self.assertRaises(SchemaError):
                parse_cot_refine_label(response)

    def test_rejects_text_after_anchor_wrong_section_and_missing_rationale(self) -> None:
        invalid = (
            (
                "REASONING:\nObject.\nFINAL_LABEL_FOR_ORIGINAL_QUESTION: True\n"
                "postscript"
            ),
            "REVIEW:\nObject.\nFINAL_LABEL_FOR_ORIGINAL_QUESTION: True",
            "REASONING:\n\nFINAL_LABEL_FOR_ORIGINAL_QUESTION: True",
            "preface\nREASONING:\nObject.\nFINAL_LABEL_FOR_ORIGINAL_QUESTION: True",
            "```\nREASONING:\nObject.\nFINAL_LABEL_FOR_ORIGINAL_QUESTION: True\n```",
        )
        for response in invalid:
            with self.subTest(response=response), self.assertRaises(SchemaError):
                parse_cot_label(response)

    def test_trailing_blank_lines_are_allowed_but_no_full_text_fallback_exists(self) -> None:
        response = self._response("REASONING", "Unknown") + "\n\n"
        self.assertEqual(parse_cot_label(response), "Unknown")
        with self.assertRaises(SchemaError):
            parse_cot_label("Object-level conclusion Unknown. Then an isolated True")

    def test_stray_rationale_labels_cannot_override_the_anchor(self) -> None:
        cot = (
            "REASONING:\nThe object-level conclusion is Unknown.\n"
            "A stray formatting artifact says True.\n\n"
            "FINAL_LABEL_FOR_ORIGINAL_QUESTION: Unknown"
        )
        refine = cot.replace("REASONING:", "REVIEW:")
        self.assertEqual(parse_cot_label(cot), "Unknown")
        self.assertEqual(parse_cot_refine_label(refine), "Unknown")

    def test_generic_direct_parser_remains_lenient(self) -> None:
        self.assertEqual(parse_label("First True, after checking: Unknown"), "Unknown")

    def test_method_failure_never_falls_back_to_last_label_regex(self) -> None:
        root = Path(__file__).resolve().parents[1]
        runtime = root / "outputs/test_runtime/patch_refine_contract"
        runtime.mkdir(parents=True, exist_ok=True)
        config = ClientConfig(
            provider="test",
            model_id="strict-refine-test",
            cache_dir=str(runtime / "cache"),
            raw_dir=str(runtime / "raw"),
            use_cache=False,
            reasoning_effort="none",
        )
        runner = MethodRunner(BadRefineClient(config), root / "configs/prompts")
        example = Example(
            "contract:Q1",
            0,
            (SourceSentence("s1", "Alice is red."),),
            SourceSentence("q1", "Alice is blue."),
            "Unknown",
            "test",
        )
        refine = runner.run("cot_refine", example.model_view())
        cot = runner.run("cot", example.model_view())
        self.assertEqual(refine.predicted_label, "Error")
        self.assertTrue(refine.method_error)
        self.assertFalse(refine.infrastructure_error)
        self.assertFalse(refine.cot_refine_contract_valid)
        self.assertEqual(cot.predicted_label, "Unknown")
        self.assertFalse(cot.method_error)
        self.assertIsNone(cot.cot_refine_contract_valid)

    def test_gate_fallback_contract_failure_is_method_error(self) -> None:
        root = Path(__file__).resolve().parents[1]
        runtime = root / "outputs/test_runtime/patch_cot_fallback_contract"
        runtime.mkdir(parents=True, exist_ok=True)
        config = ClientConfig(
            provider="test",
            model_id="bad-cot-fallback-test",
            cache_dir=str(runtime / "cache"),
            raw_dir=str(runtime / "raw"),
            use_cache=False,
            reasoning_effort="none",
        )
        runner = MethodRunner(BadCotFallbackClient(config), root / "configs/prompts")
        example = Example(
            "fallback-contract:Q1",
            0,
            (SourceSentence("s1", "Alice is red."),),
            SourceSentence("q1", "Alice is blue."),
            "Unknown",
            "test",
        )
        output = runner.run("gate_cot", example.model_view())
        self.assertEqual(output.predicted_label, "Error")
        self.assertTrue(output.method_error)
        self.assertTrue(output.fallback_used)
        self.assertFalse(output.infrastructure_error)
        self.assertIn("output contract failed", output.error or "")


class SingletonIfTests(unittest.TestCase):
    def _value(self, premise: Any) -> dict[str, Any]:
        return {
            "facts": [{"id": "s1", "atom": "+blue(alice)"}],
            "rules": [{"id": "s2", "if": premise, "then": "+warm(X)"}],
            "query": {"id": "q1", "atom": "+warm(alice)"},
        }

    def test_wraps_only_complete_signed_unary_or_binary_atoms(self) -> None:
        valid = ("+blue(X)", "-blue(X)", "+needs(X,the_cat)", "-needs(X,the_cat)")
        for premise in valid:
            with self.subTest(premise=premise):
                value = self._value(premise)
                normalized, count = normalize_singleton_rule_if(value)
                self.assertEqual(count, 1)
                self.assertEqual(normalized["rules"][0]["if"], [premise])
                self.assertEqual(value["rules"][0]["if"], premise)

    def test_forbidden_strings_are_exact_noops(self) -> None:
        forbidden = (
            "+p(a),+q(b)",
            "+p(a) and +q(b)",
            "+p(a b)",
            "+p()",
            "p(a)",
            "+p(a)->+q(a)",
            "",
            "natural language premise",
        )
        for premise in forbidden:
            with self.subTest(premise=premise):
                value = self._value(premise)
                normalized, count = normalize_singleton_rule_if(value)
                self.assertIs(normalized, value)
                self.assertEqual(count, 0)

    def test_existing_array_is_identity_noop(self) -> None:
        value = self._value(["+blue(X)"])
        normalized, count = normalize_singleton_rule_if(value)
        self.assertIs(normalized, value)
        self.assertEqual(count, 0)

    def test_strict_and_normalized_schema_are_both_retained(self) -> None:
        raw = json.dumps(self._value("+blue(X)"))
        with self.assertRaises(SchemaError):
            parse_v2_ir_strict(raw)
        self.assertEqual(parse_v2_ir(raw).rules[0].premises[0].render(), "+blue(X)")
        example = Example(
            "normalization:Q1",
            1,
            (
                SourceSentence("s1", "Alice is blue."),
                SourceSentence("s2", "If someone is blue then they are warm."),
            ),
            SourceSentence("q1", "Alice is warm."),
            "True",
            "test",
        )
        result = StaticValidator().validate_json(raw, example)
        self.assertTrue(result.json_parse_valid)
        self.assertFalse(result.strict_schema_valid)
        self.assertTrue(result.normalized_schema_valid)
        self.assertTrue(result.schema_valid)
        self.assertEqual(result.singleton_if_normalized_count, 1)


if __name__ == "__main__":
    unittest.main()
