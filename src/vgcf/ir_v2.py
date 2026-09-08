"""Compact external JSON IR used by every VGCF-2 formalization method.

The model-facing representation is deliberately smaller than the internal
dataclass schema.  Parsing is deterministic: it never invents predicates,
signs, arguments, or rule directions.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

from .errors import SchemaError
from .schema import Atom, Formalization, Rule, Term

IR_CONTRACT_ID = "vgcf-ir-v2"

_IDENTIFIER = r"[A-Za-z][A-Za-z0-9_]*"
_PREDICATE = r"[a-z][a-z0-9_]*"
_ATOM_PATTERN = re.compile(
    rf"^(?P<sign>[+-])(?P<predicate>{_PREDICATE})"
    rf"\((?P<arguments>{_IDENTIFIER}(?:\s*,\s*{_IDENTIFIER})*)\)$"
)
_FENCE_PATTERN = re.compile(
    r"^\s*```(?:json)?\s*\r?\n?(?P<body>.*?)\r?\n?```\s*$",
    flags=re.DOTALL | re.IGNORECASE,
)


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaError(f"{path} must be a JSON object")
    return value


def _keys(value: Mapping[str, Any], required: set[str], path: str) -> None:
    missing = required - set(value)
    unknown = set(value) - required
    if missing:
        raise SchemaError(f"{path} missing required fields: {sorted(missing)}")
    if unknown:
        raise SchemaError(f"{path} has unknown fields: {sorted(unknown)}")


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{path} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True, slots=True)
class V2Atom:
    """A signed unary or binary atom in compact textual form."""

    predicate: str
    arguments: tuple[str, ...]
    negated: bool

    @classmethod
    def parse(cls, value: Any, path: str) -> "V2Atom":
        text = _text(value, path)
        match = _ATOM_PATTERN.fullmatch(text)
        if match is None:
            raise SchemaError(
                f"{path} must match +predicate(arg) or -predicate(arg1,arg2) "
                "using identifiers without spaces"
            )
        arguments = tuple(
            part.strip() for part in match.group("arguments").split(",")
        )
        if len(arguments) not in {1, 2}:
            raise SchemaError(f"{path} must have exactly one or two arguments")
        return cls(
            predicate=match.group("predicate"),
            arguments=arguments,
            negated=match.group("sign") == "-",
        )

    def to_atom(
        self, source_id: str | None = None, *, allow_variables: bool = True
    ) -> Atom:
        """Convert without semantic guessing.

        In ground positions, argument case is deterministically normalized to
        lowercase. Inside rules, an uppercase initial retains its documented
        variable meaning and lowercase arguments remain constants.
        """

        terms = tuple(
            Term(
                name=(
                    argument
                    if allow_variables or argument.isupper()
                    else argument.lower()
                ),
                kind=(
                    "variable"
                    if (allow_variables and argument[0].isupper())
                    or (not allow_variables and argument.isupper())
                    else "constant"
                ),
            )
            for argument in self.arguments
        )
        return Atom(self.predicate, terms, self.negated, source_id)

    def render(self) -> str:
        sign = "-" if self.negated else "+"
        return f"{sign}{self.predicate}({','.join(self.arguments)})"


@dataclass(frozen=True, slots=True)
class V2Fact:
    source_id: str
    atom: V2Atom

    @classmethod
    def from_dict(cls, value: Any, path: str) -> "V2Fact":
        obj = _mapping(value, path)
        _keys(obj, {"id", "atom"}, path)
        return cls(_text(obj["id"], f"{path}.id"), V2Atom.parse(obj["atom"], f"{path}.atom"))


@dataclass(frozen=True, slots=True)
class V2Rule:
    source_id: str
    premises: tuple[V2Atom, ...]
    conclusion: V2Atom

    @classmethod
    def from_dict(cls, value: Any, path: str) -> "V2Rule":
        obj = _mapping(value, path)
        _keys(obj, {"id", "if", "then"}, path)
        premises = obj["if"]
        if not isinstance(premises, list) or not premises:
            raise SchemaError(f"{path}.if must be a non-empty JSON array")
        return cls(
            source_id=_text(obj["id"], f"{path}.id"),
            premises=tuple(
                V2Atom.parse(atom, f"{path}.if[{index}]")
                for index, atom in enumerate(premises)
            ),
            conclusion=V2Atom.parse(obj["then"], f"{path}.then"),
        )


@dataclass(frozen=True, slots=True)
class V2Query:
    source_id: str
    atom: V2Atom

    @classmethod
    def from_dict(cls, value: Any, path: str = "$.query") -> "V2Query":
        obj = _mapping(value, path)
        _keys(obj, {"id", "atom"}, path)
        return cls(_text(obj["id"], f"{path}.id"), V2Atom.parse(obj["atom"], f"{path}.atom"))


@dataclass(frozen=True, slots=True)
class V2IR:
    facts: tuple[V2Fact, ...]
    rules: tuple[V2Rule, ...]
    query: V2Query

    @classmethod
    def from_dict(cls, value: Any) -> "V2IR":
        obj = _mapping(value, "$")
        _keys(obj, {"facts", "rules", "query"}, "$")
        if not isinstance(obj["facts"], list):
            raise SchemaError("$.facts must be a JSON array")
        if not isinstance(obj["rules"], list):
            raise SchemaError("$.rules must be a JSON array")
        return cls(
            facts=tuple(
                V2Fact.from_dict(item, f"$.facts[{index}]")
                for index, item in enumerate(obj["facts"])
            ),
            rules=tuple(
                V2Rule.from_dict(item, f"$.rules[{index}]")
                for index, item in enumerate(obj["rules"])
            ),
            query=V2Query.from_dict(obj["query"]),
        )

    def to_formalization(self) -> Formalization:
        return Formalization(
            facts=tuple(
                item.atom.to_atom(item.source_id, allow_variables=False)
                for item in self.facts
            ),
            rules=tuple(
                Rule(
                    tuple(atom.to_atom() for atom in item.premises),
                    item.conclusion.to_atom(),
                    item.source_id,
                )
                for item in self.rules
            ),
            query=self.query.atom.to_atom(
                self.query.source_id, allow_variables=False
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "facts": [
                {"id": item.source_id, "atom": item.atom.render()}
                for item in self.facts
            ],
            "rules": [
                {
                    "id": item.source_id,
                    "if": [atom.render() for atom in item.premises],
                    "then": item.conclusion.render(),
                }
                for item in self.rules
            ],
            "query": {
                "id": self.query.source_id,
                "atom": self.query.atom.render(),
            },
        }


def strip_optional_fence(text: str) -> str:
    """Remove one complete Markdown JSON fence, but never surrounding prose."""

    match = _FENCE_PATTERN.fullmatch(text)
    return match.group("body").strip() if match else text.strip()


def parse_v2_json(text: str) -> Any:
    """Parse one complete JSON value after optional-fence removal."""

    try:
        return json.loads(strip_optional_fence(text))
    except json.JSONDecodeError as exc:
        raise SchemaError(f"response is not valid JSON: {exc}") from exc


def normalize_singleton_rule_if(value: Any) -> tuple[Any, int]:
    """Safely wrap complete singleton atoms found at ``rules[*].if`` only.

    The transformation is intentionally narrower than schema repair. It never
    splits strings, moves objects, edits atoms, or mutates the caller's value.
    If no eligible string exists, the exact original object is returned.
    """

    if not isinstance(value, Mapping):
        return value, 0
    rules = value.get("rules")
    if not isinstance(rules, list):
        return value, 0
    eligible: list[int] = []
    for index, rule in enumerate(rules):
        if not isinstance(rule, Mapping):
            continue
        premise = rule.get("if")
        if isinstance(premise, str) and _ATOM_PATTERN.fullmatch(premise):
            eligible.append(index)
    if not eligible:
        return value, 0
    normalized = deepcopy(value)
    normalized_rules = normalized["rules"]
    for index in eligible:
        premise = normalized_rules[index]["if"]
        normalized_rules[index]["if"] = [premise]
    return normalized, len(eligible)


def parse_v2_ir(text: str, *, normalize_singleton_if: bool = True) -> V2IR:
    value = parse_v2_json(text)
    if normalize_singleton_if:
        value, _ = normalize_singleton_rule_if(value)
    return V2IR.from_dict(value)


def parse_v2_ir_strict(text: str) -> V2IR:
    """Parse the raw model object without singleton-if normalization."""

    return parse_v2_ir(text, normalize_singleton_if=False)


def parse_v2_formalization(text: str) -> Formalization:
    return parse_v2_ir(text).to_formalization()


def formalization_to_v2_dict(formalization: Formalization) -> dict[str, Any]:
    """Serialize an internal program for Mock/tests, not as an auto-repair path."""

    def render(atom: Atom) -> str:
        sign = "-" if atom.negated else "+"
        arguments = [
            (term.name[:1].upper() + term.name[1:])
            if term.kind == "variable"
            else term.name
            for term in atom.arguments
        ]
        return f"{sign}{atom.predicate}({','.join(arguments)})"

    return {
        "facts": [
            {"id": fact.source_id, "atom": render(fact)}
            for fact in formalization.facts
        ],
        "rules": [
            {
                "id": rule.source_id,
                "if": [render(atom) for atom in rule.premises],
                "then": render(rule.conclusion),
            }
            for rule in formalization.rules
        ],
        "query": {
            "id": formalization.query.source_id,
            "atom": render(formalization.query),
        },
    }
