"""A tiny transparent parser used only by fixtures and the deterministic Mock.

This is not presented as a general ProofWriter semantic parser. Real language
models replace it in experiments; keeping it narrow makes mock behavior honest
and reproducible.
"""

from __future__ import annotations

import re
from typing import Sequence

from .errors import SchemaError
from .schema import Atom, Formalization, Rule, Term

_VERB_LEMMAS = {
    "likes": "like",
    "sees": "see",
    "chases": "chase",
    "visits": "visit",
    "admires": "admire",
    "needs": "need",
}
_VARIABLE_WORDS = {"someone", "they", "them"}


def _term(token: str) -> Term:
    cleaned = token.strip().lower()
    if cleaned in _VARIABLE_WORDS:
        return Term("x", "variable")
    return Term(cleaned, "constant")


def parse_atom_text(text: str, source_id: str | None = None) -> Atom:
    cleaned = text.strip().rstrip(".?").strip()
    unary = re.fullmatch(
        r"(?P<subject>[A-Za-z]+)\s+(?:is|are)\s+(?P<negated>not\s+)?(?P<predicate>[a-z]+)",
        cleaned,
        flags=re.IGNORECASE,
    )
    if unary:
        return Atom(
            predicate=unary.group("predicate").lower(),
            arguments=(_term(unary.group("subject")),),
            negated=bool(unary.group("negated")),
            source_id=source_id,
        )
    binary = re.fullmatch(
        r"(?P<subject>[A-Za-z]+)\s+"
        r"(?:(?P<negative>(?:does|do)\s+not)\s+)?"
        r"(?P<predicate>[a-z]+)\s+(?P<object>[A-Za-z]+)",
        cleaned,
        flags=re.IGNORECASE,
    )
    if binary:
        predicate = binary.group("predicate").lower()
        predicate = _VERB_LEMMAS.get(predicate, predicate)
        return Atom(
            predicate=predicate,
            arguments=(_term(binary.group("subject")), _term(binary.group("object"))),
            negated=bool(binary.group("negative")),
            source_id=source_id,
        )
    raise SchemaError(f"controlled-language parser cannot parse atom: {text!r}")


def parse_theory_sentence(text: str, source_id: str) -> Atom | Rule:
    cleaned = text.strip()
    if cleaned.lower().startswith("if "):
        body = cleaned.rstrip(".")[3:]
        pieces = re.split(r"\s+then\s+", body, maxsplit=1, flags=re.IGNORECASE)
        if len(pieces) != 2:
            raise SchemaError(f"rule has no single 'then': {text!r}")
        premise_text, conclusion_text = pieces
        premise_parts = re.split(r"\s+and\s+", premise_text, flags=re.IGNORECASE)
        premises = tuple(parse_atom_text(part) for part in premise_parts)
        return Rule(
            premises=premises,
            conclusion=parse_atom_text(conclusion_text),
            source_id=source_id,
        )
    return parse_atom_text(cleaned, source_id=source_id)


def formalize_controlled(
    theory: Sequence[str], question: str
) -> Formalization:
    facts: list[Atom] = []
    rules: list[Rule] = []
    for index, text in enumerate(theory, start=1):
        structure = parse_theory_sentence(text, f"s{index}")
        if isinstance(structure, Rule):
            rules.append(structure)
        else:
            facts.append(structure)
    query = parse_atom_text(question, source_id="q1")
    if any(term.kind == "variable" for term in query.arguments):
        raise SchemaError("fixture questions must be ground")
    return Formalization(tuple(facts), tuple(rules), query)
