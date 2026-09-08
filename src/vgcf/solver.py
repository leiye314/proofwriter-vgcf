"""Deterministic Datalog-style forward chaining under open-world semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .errors import LogicError
from .schema import Atom, Formalization, Rule


@dataclass(frozen=True, order=True, slots=True)
class GroundAtom:
    predicate: str
    arguments: tuple[str, ...]
    negated: bool = False

    def complement(self) -> "GroundAtom":
        return GroundAtom(self.predicate, self.arguments, not self.negated)

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicate": self.predicate,
            "arguments": list(self.arguments),
            "negated": self.negated,
        }

    def __str__(self) -> str:
        prefix = "not " if self.negated else ""
        return f"{prefix}{self.predicate}({', '.join(self.arguments)})"


@dataclass(frozen=True, slots=True)
class Derivation:
    fact: GroundAtom
    kind: str
    source_id: str
    premises: tuple[GroundAtom, ...] = ()
    binding: tuple[tuple[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact": self.fact.to_dict(),
            "kind": self.kind,
            "source_id": self.source_id,
            "premises": [premise.to_dict() for premise in self.premises],
            "binding": dict(self.binding),
        }


@dataclass(frozen=True, slots=True)
class SolveResult:
    label: str
    query: GroundAtom
    target: GroundAtom | None
    closure: tuple[GroundAtom, ...]
    proof: tuple[Derivation, ...]
    iterations: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "query": self.query.to_dict(),
            "target": self.target.to_dict() if self.target else None,
            "proof": [step.to_dict() for step in self.proof],
            "iterations": self.iterations,
            "closure_size": len(self.closure),
        }


class ForwardChainingSolver:
    """Apply rules only in their stated direction until a fixed point.

    Absence from the closure means ``Unknown`` unless the explicit complement
    is derivable. No closed-world assumption, inverse rule, or contraposition is
    implemented.
    """

    def __init__(self, max_iterations: int = 1000) -> None:
        self.max_iterations = max_iterations

    def solve(self, formalization: Formalization) -> SolveResult:
        query = self._ground_atom(formalization.query, {})
        closure: set[GroundAtom] = set()
        derivations: dict[GroundAtom, Derivation] = {}

        for fact in formalization.facts:
            ground = self._ground_atom(fact, {})
            if ground not in closure:
                closure.add(ground)
                derivations[ground] = Derivation(
                    fact=ground,
                    kind="given",
                    source_id=fact.source_id or "missing-source",
                )

        iterations = 0
        changed = True
        while changed:
            if iterations >= self.max_iterations:
                raise LogicError(
                    f"forward chaining exceeded {self.max_iterations} iterations"
                )
            changed = False
            iterations += 1
            snapshot = tuple(sorted(closure))
            for rule in formalization.rules:
                for binding, matched in self._match_rule(rule, snapshot):
                    conclusion = self._ground_atom(rule.conclusion, binding)
                    if conclusion in closure:
                        continue
                    closure.add(conclusion)
                    derivations[conclusion] = Derivation(
                        fact=conclusion,
                        kind="rule",
                        source_id=rule.source_id,
                        premises=matched,
                        binding=tuple(sorted(binding.items())),
                    )
                    changed = True

        complement = query.complement()
        if query in closure and complement in closure:
            raise LogicError(
                f"inconsistent theory derives both {query} and {complement}"
            )
        if query in closure:
            label, target = "True", query
        elif complement in closure:
            label, target = "False", complement
        else:
            label, target = "Unknown", None
        proof = self._collect_proof(target, derivations) if target else ()
        return SolveResult(
            label=label,
            query=query,
            target=target,
            closure=tuple(sorted(closure)),
            proof=proof,
            iterations=iterations,
        )

    @staticmethod
    def _ground_atom(atom: Atom, binding: Mapping[str, str]) -> GroundAtom:
        arguments: list[str] = []
        for term in atom.arguments:
            if term.kind == "constant":
                arguments.append(term.name)
            elif term.name in binding:
                arguments.append(binding[term.name])
            else:
                raise LogicError(
                    f"unbound variable {term.name!r} in {atom.predicate} conclusion/query"
                )
        return GroundAtom(atom.predicate, tuple(arguments), atom.negated)

    def _match_rule(
        self, rule: Rule, facts: Iterable[GroundAtom]
    ) -> Iterable[tuple[dict[str, str], tuple[GroundAtom, ...]]]:
        fact_list = tuple(facts)

        def visit(
            premise_index: int,
            binding: dict[str, str],
            matched: tuple[GroundAtom, ...],
        ) -> Iterable[tuple[dict[str, str], tuple[GroundAtom, ...]]]:
            if premise_index == len(rule.premises):
                yield dict(binding), matched
                return
            pattern = rule.premises[premise_index]
            for fact in fact_list:
                next_binding = self._unify(pattern, fact, binding)
                if next_binding is not None:
                    yield from visit(
                        premise_index + 1,
                        next_binding,
                        matched + (fact,),
                    )

        return visit(0, {}, ())

    @staticmethod
    def _unify(
        pattern: Atom, fact: GroundAtom, binding: Mapping[str, str]
    ) -> dict[str, str] | None:
        if (
            pattern.predicate != fact.predicate
            or pattern.negated != fact.negated
            or len(pattern.arguments) != len(fact.arguments)
        ):
            return None
        result = dict(binding)
        for term, value in zip(pattern.arguments, fact.arguments):
            if term.kind == "constant":
                if term.name != value:
                    return None
            elif term.name in result:
                if result[term.name] != value:
                    return None
            else:
                result[term.name] = value
        return result

    @staticmethod
    def _collect_proof(
        target: GroundAtom, derivations: Mapping[GroundAtom, Derivation]
    ) -> tuple[Derivation, ...]:
        ordered: list[Derivation] = []
        visited: set[GroundAtom] = set()

        def visit(fact: GroundAtom) -> None:
            if fact in visited:
                return
            visited.add(fact)
            derivation = derivations[fact]
            for premise in derivation.premises:
                visit(premise)
            ordered.append(derivation)

        visit(target)
        return tuple(ordered)
