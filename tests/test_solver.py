from __future__ import annotations

import unittest

from vgcf.schema import Formalization, Rule
from vgcf.solver import ForwardChainingSolver

from tests.helpers import atom, constant, variable


class SolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.solver = ForwardChainingSolver()
        self.alice = constant("alice")
        self.bob = constant("bob")
        self.x = variable()

    def test_true_false_unknown_open_world(self) -> None:
        true_program = Formalization(
            (atom("red", self.alice, source_id="s1"),),
            (),
            atom("red", self.alice, source_id="q1"),
        )
        false_program = Formalization(
            (atom("red", self.alice, negated=True, source_id="s1"),),
            (),
            atom("red", self.alice, source_id="q1"),
        )
        unknown_program = Formalization(
            (atom("blue", self.alice, source_id="s1"),),
            (),
            atom("red", self.alice, source_id="q1"),
        )
        self.assertEqual(self.solver.solve(true_program).label, "True")
        self.assertEqual(self.solver.solve(false_program).label, "False")
        self.assertEqual(self.solver.solve(unknown_program).label, "Unknown")

    def test_explicit_negation_can_be_derived(self) -> None:
        program = Formalization(
            (atom("cold", self.alice, source_id="s1"),),
            (
                Rule(
                    (atom("cold", self.x),),
                    atom("warm", self.x, negated=True),
                    "s2",
                ),
            ),
            atom("warm", self.alice, source_id="q1"),
        )
        result = self.solver.solve(program)
        self.assertEqual(result.label, "False")
        self.assertEqual([step.kind for step in result.proof], ["given", "rule"])

    def test_multi_premise_rule_requires_consistent_binding(self) -> None:
        program = Formalization(
            (
                atom("red", self.alice, source_id="s1"),
                atom("young", self.alice, source_id="s2"),
                atom("red", self.bob, source_id="s3"),
            ),
            (
                Rule(
                    (atom("red", self.x), atom("young", self.x)),
                    atom("kind", self.x),
                    "s4",
                ),
            ),
            atom("kind", self.bob, source_id="q1"),
        )
        self.assertEqual(self.solver.solve(program).label, "Unknown")
        alice_query = Formalization(program.facts, program.rules, atom("kind", self.alice, source_id="q1"))
        self.assertEqual(self.solver.solve(alice_query).label, "True")

    def test_binary_argument_order_is_preserved(self) -> None:
        program = Formalization(
            (atom("like", self.alice, self.bob, source_id="s1"),),
            (),
            atom("like", self.bob, self.alice, source_id="q1"),
        )
        self.assertEqual(self.solver.solve(program).label, "Unknown")

    def test_no_inverse_or_contraposition(self) -> None:
        rule = Rule((atom("red", self.x),), atom("warm", self.x), "s1")
        inverse = Formalization(
            (atom("warm", self.alice, source_id="s2"),),
            (rule,),
            atom("red", self.alice, source_id="q1"),
        )
        contraposition = Formalization(
            (atom("warm", self.alice, negated=True, source_id="s2"),),
            (rule,),
            atom("red", self.alice, negated=True, source_id="q1"),
        )
        self.assertEqual(self.solver.solve(inverse).label, "Unknown")
        self.assertEqual(self.solver.solve(contraposition).label, "Unknown")


if __name__ == "__main__":
    unittest.main()
