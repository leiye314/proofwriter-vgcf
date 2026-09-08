from __future__ import annotations

import unittest

from vgcf.gold import semantic_metrics
from vgcf.schema import Formalization, Rule

from tests.helpers import atom, constant, variable


def _fact_program(predicate: str, name: str, *, negated: bool = False) -> Formalization:
    return Formalization(
        (atom(predicate, constant(name), negated=negated, source_id="s1"),),
        (),
        atom(predicate, constant(name), negated=negated, source_id="q1"),
    )


class RenamingInvariantSemanticTests(unittest.TestCase):
    def test_global_cat_to_the_cat_bijection_passes(self) -> None:
        predicted = _fact_program("blue", "the_cat")
        gold = _fact_program("blue", "cat")
        metrics = semantic_metrics(predicted, gold)
        self.assertFalse(metrics["strict_semantic_match"])
        self.assertTrue(metrics["renaming_invariant_semantic_match"])
        self.assertEqual(metrics["renaming_invariant_semantic_precision"], 1.0)
        self.assertEqual(metrics["renaming_invariant_semantic_recall"], 1.0)
        self.assertEqual(
            metrics["renaming_invariant_constant_mapping"], {"the_cat": "cat"}
        )

    def test_one_constant_cannot_map_to_two_objects(self) -> None:
        predicted = Formalization(
            (
                atom("blue", constant("the_animal"), source_id="s1"),
                atom("young", constant("the_animal"), source_id="s2"),
            ),
            (),
            atom("blue", constant("the_animal"), source_id="q1"),
        )
        gold = Formalization(
            (
                atom("blue", constant("cat"), source_id="s1"),
                atom("young", constant("dog"), source_id="s2"),
            ),
            (),
            atom("blue", constant("cat"), source_id="q1"),
        )
        self.assertFalse(
            semantic_metrics(predicted, gold)["renaming_invariant_semantic_match"]
        )

    def test_two_constants_cannot_map_to_one_object(self) -> None:
        predicted = Formalization(
            (
                atom("blue", constant("the_cat"), source_id="s1"),
                atom("young", constant("the_dog"), source_id="s2"),
            ),
            (),
            atom("blue", constant("the_cat"), source_id="q1"),
        )
        gold = Formalization(
            (
                atom("blue", constant("animal"), source_id="s1"),
                atom("young", constant("animal"), source_id="s2"),
            ),
            (),
            atom("blue", constant("animal"), source_id="q1"),
        )
        self.assertFalse(
            semantic_metrics(predicted, gold)["renaming_invariant_semantic_match"]
        )

    def test_argument_swap_fails_when_global_constants_are_anchored(self) -> None:
        predicted = Formalization(
            (
                atom("visits", constant("cat"), constant("dog"), source_id="s1"),
                atom("blue", constant("cat"), source_id="s2"),
                atom("young", constant("dog"), source_id="s3"),
            ),
            (),
            atom("visits", constant("cat"), constant("dog"), source_id="q1"),
        )
        gold = Formalization(
            (
                atom("visits", constant("dog"), constant("cat"), source_id="s1"),
                atom("blue", constant("cat"), source_id="s2"),
                atom("young", constant("dog"), source_id="s3"),
            ),
            (),
            atom("visits", constant("dog"), constant("cat"), source_id="q1"),
        )
        self.assertFalse(
            semantic_metrics(predicted, gold)["renaming_invariant_semantic_match"]
        )

    def test_polarity_and_predicate_changes_fail(self) -> None:
        gold = _fact_program("blue", "cat")
        self.assertFalse(
            semantic_metrics(
                _fact_program("blue", "cat", negated=True), gold
            )["renaming_invariant_semantic_match"]
        )
        self.assertFalse(
            semantic_metrics(_fact_program("green", "cat"), gold)[
                "renaming_invariant_semantic_match"
            ]
        )

    def test_rule_direction_change_fails(self) -> None:
        predicted = Formalization(
            (atom("p", constant("a"), source_id="s1"),),
            (Rule((atom("p", variable("X")),), atom("q", variable("X")), "s2"),),
            atom("q", constant("a"), source_id="q1"),
        )
        gold = Formalization(
            (atom("p", constant("a"), source_id="s1"),),
            (Rule((atom("q", variable("Y")),), atom("p", variable("Y")), "s2"),),
            atom("q", constant("a"), source_id="q1"),
        )
        self.assertFalse(
            semantic_metrics(predicted, gold)["renaming_invariant_semantic_match"]
        )

    def test_variable_alpha_renaming_passes(self) -> None:
        predicted = Formalization(
            (atom("p", constant("the_cat"), source_id="s1"),),
            (
                Rule(
                    (
                        atom("p", variable("Person")),
                        atom("r", variable("Person"), constant("the_cat")),
                    ),
                    atom("q", variable("Person")),
                    "s2",
                ),
            ),
            atom("q", constant("the_cat"), source_id="q1"),
        )
        gold = Formalization(
            (atom("p", constant("cat"), source_id="s1"),),
            (
                Rule(
                    (
                        atom("p", variable("X")),
                        atom("r", variable("X"), constant("cat")),
                    ),
                    atom("q", variable("X")),
                    "s2",
                ),
            ),
            atom("q", constant("cat"), source_id="q1"),
        )
        metrics = semantic_metrics(predicted, gold)
        self.assertTrue(metrics["renaming_invariant_semantic_match"])
        self.assertFalse(metrics["strict_semantic_match"])


if __name__ == "__main__":
    unittest.main()
