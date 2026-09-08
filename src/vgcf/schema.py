"""Strict, dependency-free JSON schema for ProofWriter-style logic.

The classes intentionally reject unknown fields. This makes constrained model
outputs auditable and avoids silently accepting misspelled or invented data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .errors import SchemaError


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaError(f"{path} must be a JSON object, got {type(value).__name__}")
    return value


def _strict_keys(
    obj: Mapping[str, Any], required: set[str], optional: set[str], path: str
) -> None:
    missing = required - set(obj)
    unknown = set(obj) - required - optional
    if missing:
        raise SchemaError(f"{path} missing required fields: {sorted(missing)}")
    if unknown:
        raise SchemaError(f"{path} has unknown fields: {sorted(unknown)}")


def _nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{path} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True, slots=True)
class Term:
    """A constant or a variable."""

    name: str
    kind: str = "constant"

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise SchemaError("term.name must be a non-empty string")
        if self.kind not in {"constant", "variable"}:
            raise SchemaError("term.kind must be 'constant' or 'variable'")

    @classmethod
    def from_dict(cls, value: Any, path: str = "term") -> "Term":
        obj = _require_mapping(value, path)
        _strict_keys(obj, {"name", "kind"}, set(), path)
        return cls(
            name=_nonempty_string(obj["name"], f"{path}.name"),
            kind=_nonempty_string(obj["kind"], f"{path}.kind"),
        )

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "kind": self.kind}


@dataclass(frozen=True, slots=True)
class Atom:
    """A possibly explicitly-negated predicate application."""

    predicate: str
    arguments: tuple[Term, ...]
    negated: bool = False
    source_id: str | None = None

    def __post_init__(self) -> None:
        if not self.predicate or not self.predicate.strip():
            raise SchemaError("atom.predicate must be a non-empty string")
        if not self.arguments:
            raise SchemaError("atom.arguments must contain at least one term")
        if not all(isinstance(term, Term) for term in self.arguments):
            raise SchemaError("atom.arguments must contain Term values")
        if not isinstance(self.negated, bool):
            raise SchemaError("atom.negated must be a boolean")

    @classmethod
    def from_dict(
        cls, value: Any, path: str = "atom", require_source: bool = True
    ) -> "Atom":
        obj = _require_mapping(value, path)
        required = {"predicate", "arguments", "negated"}
        if require_source:
            required.add("source_id")
        _strict_keys(obj, required, {"source_id"} - required, path)
        arguments = obj["arguments"]
        if not isinstance(arguments, list):
            raise SchemaError(f"{path}.arguments must be a JSON array")
        source_id = obj.get("source_id")
        if source_id is not None:
            source_id = _nonempty_string(source_id, f"{path}.source_id")
        return cls(
            predicate=_nonempty_string(obj["predicate"], f"{path}.predicate"),
            arguments=tuple(
                Term.from_dict(term, f"{path}.arguments[{index}]")
                for index, term in enumerate(arguments)
            ),
            negated=obj["negated"],
            source_id=source_id,
        )

    def to_dict(self, include_source: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "predicate": self.predicate,
            "arguments": [term.to_dict() for term in self.arguments],
            "negated": self.negated,
        }
        if include_source:
            result["source_id"] = self.source_id
        return result

    @property
    def variables(self) -> set[str]:
        return {term.name for term in self.arguments if term.kind == "variable"}

    @property
    def constants(self) -> set[str]:
        return {term.name for term in self.arguments if term.kind == "constant"}

    def structural_key(self) -> tuple[Any, ...]:
        return (
            self.predicate,
            tuple((term.kind, term.name) for term in self.arguments),
            self.negated,
        )


@dataclass(frozen=True, slots=True)
class Rule:
    premises: tuple[Atom, ...]
    conclusion: Atom
    source_id: str

    def __post_init__(self) -> None:
        if not self.premises:
            raise SchemaError("rule.premises must contain at least one atom")
        if not self.source_id or not self.source_id.strip():
            raise SchemaError("rule.source_id must be a non-empty string")

    @classmethod
    def from_dict(cls, value: Any, path: str = "rule") -> "Rule":
        obj = _require_mapping(value, path)
        _strict_keys(obj, {"premises", "conclusion", "source_id"}, set(), path)
        premises = obj["premises"]
        if not isinstance(premises, list):
            raise SchemaError(f"{path}.premises must be a JSON array")
        return cls(
            premises=tuple(
                Atom.from_dict(atom, f"{path}.premises[{index}]", False)
                for index, atom in enumerate(premises)
            ),
            conclusion=Atom.from_dict(obj["conclusion"], f"{path}.conclusion", False),
            source_id=_nonempty_string(obj["source_id"], f"{path}.source_id"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "premises": [atom.to_dict(include_source=False) for atom in self.premises],
            "conclusion": self.conclusion.to_dict(include_source=False),
            "source_id": self.source_id,
        }

    def structural_key(self) -> tuple[Any, ...]:
        return (
            tuple(atom.structural_key() for atom in self.premises),
            self.conclusion.structural_key(),
        )


@dataclass(frozen=True, slots=True)
class Formalization:
    facts: tuple[Atom, ...]
    rules: tuple[Rule, ...]
    query: Atom

    @classmethod
    def from_dict(cls, value: Any) -> "Formalization":
        obj = _require_mapping(value, "formalization")
        _strict_keys(obj, {"facts", "rules", "query"}, set(), "formalization")
        if not isinstance(obj["facts"], list):
            raise SchemaError("formalization.facts must be a JSON array")
        if not isinstance(obj["rules"], list):
            raise SchemaError("formalization.rules must be a JSON array")
        return cls(
            facts=tuple(
                Atom.from_dict(atom, f"formalization.facts[{index}]", True)
                for index, atom in enumerate(obj["facts"])
            ),
            rules=tuple(
                Rule.from_dict(rule, f"formalization.rules[{index}]")
                for index, rule in enumerate(obj["rules"])
            ),
            query=Atom.from_dict(obj["query"], "formalization.query", True),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "facts": [fact.to_dict() for fact in self.facts],
            "rules": [rule.to_dict() for rule in self.rules],
            "query": self.query.to_dict(),
        }

    def all_atoms(self) -> Iterable[Atom]:
        yield from self.facts
        for rule in self.rules:
            yield from rule.premises
            yield rule.conclusion
        yield self.query
