from __future__ import annotations

from vgcf.data import Example, SourceSentence
from vgcf.schema import Atom, Formalization, Rule, Term


def constant(name: str) -> Term:
    return Term(name, "constant")


def variable(name: str = "x") -> Term:
    return Term(name, "variable")


def atom(
    predicate: str,
    *terms: Term,
    negated: bool = False,
    source_id: str | None = None,
) -> Atom:
    return Atom(predicate, tuple(terms), negated, source_id)


def example(theory: list[str], question: str, gold: str = "Unknown") -> Example:
    return Example(
        "test-1",
        0,
        tuple(SourceSentence(f"s{i}", text) for i, text in enumerate(theory, 1)),
        SourceSentence("q1", question),
        gold,
        "test",
    )
