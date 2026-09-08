from __future__ import annotations

import json
import unittest

from vgcf.schema import Formalization, Rule
from vgcf.validator import StaticValidator

from tests.helpers import atom, constant, example, variable


class ValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.validator = StaticValidator()
        self.alice = constant("alice")
        self.x = variable("x")

    def test_invalid_json_and_missing_schema_fields_are_reported(self) -> None:
        source = example(["Alice is red."], "Alice is warm.")
        invalid = self.validator.validate_json("not json", source)
        missing = self.validator.validate_json('{"facts": []}', source)
        self.assertEqual(invalid.issues[0].code, "invalid_json")
        self.assertEqual(missing.issues[0].code, "schema_error")

    def test_typical_structural_errors_are_detected(self) -> None:
        source = example(
            ["Alice is red.", "If someone is red then they are warm."],
            "Alice is warm.",
        )
        malformed = Formalization(
            (
                atom("red", constant("eve"), negated=True, source_id="s1"),
                atom("cold", self.alice, source_id="q1"),
            ),
            (
                Rule(
                    (atom("red", self.x),),
                    atom("warm", variable("y")),
                    "s1",
                ),
            ),
            atom("cold", self.alice, source_id="wrong"),
        )
        codes = {issue.code for issue in self.validator.validate(malformed, source).issues}
        self.assertTrue(
            {
                "duplicate_source_id",
                "omitted_source",
                "phantom_entity",
                "unbound_conclusion_variable",
                "query_source_mismatch",
                "query_predicate_mismatch",
                "query_added_as_fact",
                "fact_negation_mismatch",
            }
            <= codes
        )

    def test_arity_and_duplicate_detection(self) -> None:
        source = example(["Alice likes Bob.", "Alice likes Bob."], "Alice likes Bob.")
        formalization = Formalization(
            (
                atom("like", self.alice, source_id="s1"),
                atom("like", self.alice, source_id="s2"),
            ),
            (),
            atom("like", self.alice, constant("bob"), source_id="q1"),
        )
        codes = {issue.code for issue in self.validator.validate(formalization, source).issues}
        self.assertIn("arity_mismatch", codes)
        self.assertIn("duplicate_fact", codes)

    def test_valid_structure_passes(self) -> None:
        source = example(["Alice is red."], "Alice is red.", "True")
        formalization = Formalization(
            (atom("red", self.alice, source_id="s1"),),
            (),
            atom("red", self.alice, source_id="q1"),
        )
        result = self.validator.validate(formalization, source)
        self.assertTrue(result.valid, [issue.to_dict() for issue in result.issues])

    def test_lexical_and_negation_heuristics_are_soft_only(self) -> None:
        source = example(["Alice is not red."], "Alice is not warm.")
        formalization = Formalization(
            (atom("red", self.alice, source_id="s1"),),
            (),
            atom("cold", self.alice, source_id="q1"),
        )
        result = self.validator.validate(formalization, source)
        self.assertTrue(result.hard_valid)
        self.assertTrue(result.soft_issues)
        self.assertTrue(all(issue.severity == "soft" for issue in result.issues))

    def test_article_preserving_multiword_entities_are_not_phantoms(self) -> None:
        source = example(
            [
                "Bob visits the bald eagle.",
                "Charlie visits the cat.",
                "The bald eagle is not quiet.",
            ],
            "Bob visits the bald eagle.",
        )
        raw = json.dumps(
            {
                "facts": [
                    {"id": "s1", "atom": "+visits(bob,the_bald_eagle)"},
                    {"id": "s2", "atom": "+visits(charlie,the_cat)"},
                    {"id": "s3", "atom": "-quiet(the_bald_eagle)"},
                ],
                "rules": [],
                "query": {"id": "q1", "atom": "+visits(bob,the_bald_eagle)"},
            }
        )
        result = self.validator.validate_json(raw, source)
        codes = {issue.code for issue in result.issues}
        self.assertTrue(result.hard_valid)
        self.assertNotIn("phantom_entity", codes)
        self.assertNotIn("query_entity_mismatch", codes)

    def test_article_dropping_alias_is_only_a_soft_warning(self) -> None:
        source = example(
            ["Bob visits the bald eagle.", "Charlie visits the cat."],
            "Bob visits the bald eagle.",
        )
        raw = json.dumps(
            {
                "facts": [
                    {"id": "s1", "atom": "+visits(bob,bald_eagle)"},
                    {"id": "s2", "atom": "+visits(charlie,cat)"},
                ],
                "rules": [],
                "query": {"id": "q1", "atom": "+visits(bob,bald_eagle)"},
            }
        )
        result = self.validator.validate_json(raw, source)
        alias_issues = [
            issue for issue in result.issues if issue.code == "inconsistent_entity_alias"
        ]
        self.assertTrue(result.hard_valid)
        self.assertGreaterEqual(len(alias_issues), 2)
        self.assertTrue(all(issue.severity == "soft" for issue in alias_issues))


if __name__ == "__main__":
    unittest.main()
