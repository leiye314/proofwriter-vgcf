"""Evaluator-only parser for official ProofWriter logic representations.

Nothing in this module is imported by prompt construction or ``MethodRunner``.
It translates the release's documented S-expression-like representation into
the same internal program consumed by the deterministic solver.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .errors import DatasetSchemaError, LogicError
from .schema import Atom, Formalization, Rule, Term
from .solver import ForwardChainingSolver

_VARIABLE_WORDS = frozenset({"someone", "something"})


@dataclass(frozen=True, slots=True)
class GoldProgram:
    example_id: str
    theory_id: str
    gold_label: str
    formalization: Formalization


def _canonical(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_]+", "_", value.strip()).strip("_").lower()
    if not normalized:
        raise DatasetSchemaError(f"empty identifier after normalizing {value!r}")
    return normalized


def _term(value: str, *, in_rule: bool) -> Term:
    if in_rule and value.lower() in _VARIABLE_WORDS:
        return Term("X", "variable")
    return Term(_canonical(value), "constant")


def _tokenize(text: str) -> list[str | tuple[str, str]]:
    tokens: list[str | tuple[str, str]] = []
    index = 0
    decoder = json.JSONDecoder()
    while index < len(text):
        if text[index].isspace():
            index += 1
            continue
        if text.startswith("->", index):
            tokens.append("->")
            index += 2
            continue
        if text[index] in "()":
            tokens.append(text[index])
            index += 1
            continue
        if text[index] == '"':
            try:
                value, consumed = decoder.raw_decode(text[index:])
            except json.JSONDecodeError as exc:
                raise DatasetSchemaError(f"invalid quoted representation token: {exc}") from exc
            if not isinstance(value, str):
                raise DatasetSchemaError("representation tokens must be strings")
            tokens.append(("string", value))
            index += consumed
            continue
        raise DatasetSchemaError(
            f"unexpected representation character {text[index]!r} at offset {index}"
        )
    return tokens


def _parse_sexpression(text: str) -> Any:
    tokens = _tokenize(text)
    cursor = 0

    def parse() -> Any:
        nonlocal cursor
        if cursor >= len(tokens):
            raise DatasetSchemaError("unexpected end of representation")
        token = tokens[cursor]
        cursor += 1
        if token == "(":
            values: list[Any] = []
            while cursor < len(tokens) and tokens[cursor] != ")":
                values.append(parse())
            if cursor >= len(tokens):
                raise DatasetSchemaError("unclosed representation parenthesis")
            cursor += 1
            return values
        if token == ")":
            raise DatasetSchemaError("unexpected closing parenthesis")
        if token == "->":
            return token
        if isinstance(token, tuple) and token[0] == "string":
            return token[1]
        raise DatasetSchemaError(f"invalid representation token {token!r}")

    value = parse()
    if cursor != len(tokens):
        raise DatasetSchemaError("trailing tokens in representation")
    return value


def _unwrap(value: Any) -> Any:
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    return value


def _is_atom_node(value: Any) -> bool:
    value = _unwrap(value)
    return (
        isinstance(value, list)
        and len(value) == 4
        and all(isinstance(item, str) for item in value)
        and value[-1] in {"+", "-", "~"}
        and value[1] != "->"
    )


def _atom_from_node(value: Any, *, in_rule: bool, source_id: str | None) -> Atom:
    node = _unwrap(value)
    if not _is_atom_node(node):
        raise DatasetSchemaError(f"expected a four-token atom, received {node!r}")
    subject, relation, obj, sign = node
    if relation == "is":
        predicate = _canonical(obj)
        arguments = (_term(subject, in_rule=in_rule),)
    else:
        predicate = _canonical(relation)
        arguments = (
            _term(subject, in_rule=in_rule),
            _term(obj, in_rule=in_rule),
        )
    # The release encodes a negative rule-body literal as ``~``.  Its README
    # explicitly states that OWA treats these as hard negation, not NAF.
    return Atom(predicate, arguments, sign in {"-", "~"}, source_id)


def parse_official_atom(representation: str, source_id: str = "q1") -> Atom:
    if not isinstance(representation, str):
        raise DatasetSchemaError("official atom representation must be a string")
    return _atom_from_node(
        _parse_sexpression(representation), in_rule=False, source_id=source_id
    )


def _premise_nodes(value: Any) -> list[Any]:
    value = _unwrap(value)
    if _is_atom_node(value):
        return [value]
    if not isinstance(value, list) or not value:
        raise DatasetSchemaError("rule premise group must contain atoms")
    result: list[Any] = []
    for item in value:
        result.extend(_premise_nodes(item))
    return result


def parse_official_rule(representation: str, source_id: str) -> Rule:
    if not isinstance(representation, str):
        raise DatasetSchemaError("official rule representation must be a string")
    node = _unwrap(_parse_sexpression(representation))
    if not isinstance(node, list) or len(node) != 3 or node[1] != "->":
        raise DatasetSchemaError(f"expected premises -> conclusion, received {node!r}")
    premises = tuple(
        _atom_from_node(item, in_rule=True, source_id=None)
        for item in _premise_nodes(node[0])
    )
    conclusion = _atom_from_node(node[2], in_rule=True, source_id=None)
    return Rule(premises, conclusion, source_id)


def _label(value: Any) -> str:
    if value is True:
        return "True"
    if value is False:
        return "False"
    if isinstance(value, str) and value.lower() == "unknown":
        return "Unknown"
    raise DatasetSchemaError(f"invalid official answer {value!r}")


def _ordered_items(value: Any, path: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, Mapping):
        raise DatasetSchemaError(f"{path} must be an object")
    result: list[Mapping[str, Any]] = []
    for key, item in value.items():
        if not isinstance(item, Mapping) or not isinstance(item.get("representation"), str):
            raise DatasetSchemaError(f"{path}.{key}.representation must be a string")
        result.append(item)
    return result


def load_gold_programs(
    path: str | Path, include_ids: Iterable[str] | None = None
) -> dict[str, GoldProgram]:
    """Load evaluator programs, optionally parsing only an explicit ID set."""

    wanted = set(include_ids) if include_ids is not None else None
    programs: dict[str, GoldProgram] = {}
    dataset_path = Path(path)
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetSchemaError(
                    f"invalid JSON at {dataset_path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(record, Mapping):
                raise DatasetSchemaError(f"record {line_number} must be an object")
            story_id = record.get("id")
            questions = record.get("questions")
            if not isinstance(story_id, str) or not isinstance(questions, Mapping):
                raise DatasetSchemaError(f"record {line_number} has invalid id/questions")
            selected_questions = [
                (question_id, item)
                for question_id, item in questions.items()
                if wanted is None or f"{story_id}:{question_id}" in wanted
            ]
            if not selected_questions:
                continue
            triples = _ordered_items(record.get("triples"), f"record[{line_number}].triples")
            rules = _ordered_items(record.get("rules"), f"record[{line_number}].rules")
            facts = tuple(
                parse_official_atom(item["representation"], f"s{index}")
                for index, item in enumerate(triples, start=1)
            )
            parsed_rules = tuple(
                parse_official_rule(item["representation"], f"s{len(facts) + index}")
                for index, item in enumerate(rules, start=1)
            )
            for question_id, item in selected_questions:
                if not isinstance(item, Mapping):
                    raise DatasetSchemaError(
                        f"record[{line_number}].questions.{question_id} must be an object"
                    )
                example_id = f"{story_id}:{question_id}"
                query = parse_official_atom(item.get("representation"), "q1")
                programs[example_id] = GoldProgram(
                    example_id=example_id,
                    theory_id=story_id,
                    gold_label=_label(item.get("answer")),
                    formalization=Formalization(facts, parsed_rules, query),
                )
    if wanted is not None and set(programs) != wanted:
        missing = sorted(wanted - set(programs))
        raise DatasetSchemaError(f"gold representations missing requested IDs: {missing[:10]}")
    return programs


def select_gold_ids(path: str | Path, count: int, seed: int) -> list[str]:
    """Choose reproducibly with at most one question from each theory."""

    if count < 0:
        raise ValueError("count must be non-negative")
    candidates: list[tuple[str, str]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, Mapping):
                raise DatasetSchemaError(f"record {line_number} must be an object")
            story_id = record.get("id")
            questions = record.get("questions")
            if not isinstance(story_id, str) or not isinstance(questions, Mapping) or not questions:
                raise DatasetSchemaError(f"record {line_number} has invalid id/questions")
            question_id = min(
                (str(key) for key in questions),
                key=lambda key: hashlib.sha256(
                    f"{seed}:{story_id}:{key}".encode("utf-8")
                ).hexdigest(),
            )
            candidates.append((story_id, f"{story_id}:{question_id}"))
    candidates.sort(
        key=lambda pair: hashlib.sha256(
            f"{seed}:{pair[0]}".encode("utf-8")
        ).hexdigest()
    )
    if count > len(candidates):
        raise ValueError(f"requested {count} theories but only {len(candidates)} exist")
    return [example_id for _, example_id in candidates[:count]]


def gold_solver_sanity(path: str | Path, count: int, seed: int) -> dict[str, Any]:
    selected_ids = select_gold_ids(path, count, seed)
    programs = load_gold_programs(path, selected_ids)
    solver = ForwardChainingSolver()
    mismatches: list[dict[str, str]] = []
    for example_id in selected_ids:
        program = programs[example_id]
        try:
            predicted = solver.solve(program.formalization).label
        except LogicError as exc:
            predicted = "Error"
            error = str(exc)
        else:
            error = ""
        if predicted != program.gold_label:
            mismatches.append(
                {
                    "example_id": example_id,
                    "gold_label": program.gold_label,
                    "solver_label": predicted,
                    "error": error,
                }
            )
    matched = count - len(mismatches)
    return {
        "dataset_path": str(Path(path).resolve()),
        "sample_count": count,
        "seed": seed,
        "one_question_per_theory": True,
        "selected_ids_sha256": hashlib.sha256(
            "\n".join(selected_ids).encode("utf-8")
        ).hexdigest(),
        "matched": matched,
        "mismatched": len(mismatches),
        "reproduction_rate": matched / count if count else 0.0,
        "threshold": 0.99,
        "passes_threshold": bool(count and matched / count >= 0.99),
        "mismatches": mismatches,
    }


def semantic_metrics(
    predicted: Formalization, gold: Formalization
) -> dict[str, Any]:
    """Return strict and evaluator-only renaming-invariant semantic metrics.

    Neither diagnostic stems or merges predicates. The renaming-invariant
    comparator additionally permits one globally consistent bijection between
    predicted and gold constants while preserving every structural invariant.
    """

    def atom_key(atom: Atom) -> tuple[Any, ...]:
        return atom.structural_key()

    def units(formalization: Formalization) -> set[tuple[Any, ...]]:
        result: set[tuple[Any, ...]] = set()
        for fact in formalization.facts:
            result.add(("fact", fact.source_id, atom_key(fact)))
        for rule in formalization.rules:
            result.add(
                (
                    "rule",
                    rule.source_id,
                    frozenset(atom_key(atom) for atom in rule.premises),
                    atom_key(rule.conclusion),
                )
            )
        result.add(("query", formalization.query.source_id, atom_key(formalization.query)))
        return result

    predicted_units = units(predicted)
    gold_units = units(gold)
    exact = _set_metrics(predicted_units, gold_units)
    predicted_normalized = _normalized_units(predicted)
    gold_normalized = _normalized_units(gold)
    normalized = _set_metrics(predicted_normalized, gold_normalized)
    renaming = _renaming_invariant_metrics(predicted, gold)
    return {
        "strict_semantic_match": exact["match"],
        "strict_semantic_precision": exact["precision"],
        "strict_semantic_recall": exact["recall"],
        "strict_semantic_f1": exact["f1"],
        "renaming_invariant_semantic_match": renaming["match"],
        "renaming_invariant_semantic_precision": renaming["precision"],
        "renaming_invariant_semantic_recall": renaming["recall"],
        "renaming_invariant_semantic_f1": renaming["f1"],
        "renaming_invariant_constant_mapping": renaming["constant_mapping"],
        "renaming_invariant_mapping_is_bijective": renaming["mapping_is_bijective"],
        "exact_match": exact["match"],
        "exact_precision": exact["precision"],
        "exact_recall": exact["recall"],
        "exact_f1": exact["f1"],
        # Backward-compatible strict aliases.
        "precision": exact["precision"],
        "recall": exact["recall"],
        "f1": exact["f1"],
        "normalized_diagnostic_match": normalized["match"],
        "normalized_diagnostic_precision": normalized["precision"],
        "normalized_diagnostic_recall": normalized["recall"],
        "normalized_diagnostic_f1": normalized["f1"],
        "exact_match_interpretation": "lower_bound",
        "strict_semantic_interpretation": "literal_lower_bound",
    }


def _set_metrics(
    predicted: set[tuple[Any, ...]], gold: set[tuple[Any, ...]]
) -> dict[str, float | bool]:
    overlap = len(predicted & gold)
    precision = overlap / len(predicted) if predicted else 0.0
    recall = overlap / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "match": predicted == gold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _renaming_invariant_metrics(
    predicted: Formalization, gold: Formalization
) -> dict[str, Any]:
    predicted_units = _source_aligned_units(predicted)
    gold_units = _source_aligned_units(gold)
    common = sorted(set(predicted_units) & set(gold_units))
    candidate_groups: list[
        tuple[tuple[str, str], list[tuple[tuple[str, str], ...]]]
    ] = []
    structurally_alignable: dict[
        tuple[str, str], list[tuple[tuple[str, str], ...]]
    ] = {}
    all_unique = all(len(items) == 1 for items in predicted_units.values()) and all(
        len(items) == 1 for items in gold_units.values()
    )
    if all_unique:
        for key in common:
            candidates = _unit_alignment_candidates(
                predicted_units[key][0], gold_units[key][0]
            )
            if candidates:
                structurally_alignable[key] = candidates
                candidate_groups.append((key, candidates))

    exact_source_shape = (
        all_unique
        and set(predicted_units) == set(gold_units)
        and len(candidate_groups) == len(predicted_units)
    )
    complete_mapping = (
        _find_complete_bijection(candidate_groups) if exact_source_shape else None
    )
    if complete_mapping is not None:
        mapping = complete_mapping
    else:
        mapping = _greedy_consistent_bijection(candidate_groups)

    matched = 0
    for key, candidates in structurally_alignable.items():
        if any(_candidate_satisfied(candidate, mapping) for candidate in candidates):
            matched += 1
    predicted_count = sum(len(items) for items in predicted_units.values())
    gold_count = sum(len(items) for items in gold_units.values())
    precision = matched / predicted_count if predicted_count else 0.0
    recall = matched / gold_count if gold_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    exact_match = exact_source_shape and complete_mapping is not None
    return {
        "match": exact_match,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "constant_mapping": dict(sorted(mapping.items())),
        "mapping_is_bijective": len(set(mapping.values())) == len(mapping),
    }


def _source_aligned_units(
    formalization: Formalization,
) -> dict[tuple[str, str], list[Atom | Rule]]:
    result: dict[tuple[str, str], list[Atom | Rule]] = {}
    for fact in formalization.facts:
        result.setdefault(("fact", fact.source_id or ""), []).append(fact)
    for rule in formalization.rules:
        result.setdefault(("rule", rule.source_id), []).append(rule)
    result.setdefault(
        ("query", formalization.query.source_id or ""), []
    ).append(formalization.query)
    return result


def _unit_alignment_candidates(
    predicted: Atom | Rule, gold: Atom | Rule
) -> list[tuple[tuple[str, str], ...]]:
    if isinstance(predicted, Atom) and isinstance(gold, Atom):
        constraints = _aligned_atom_constraints(predicted, gold, {}, {})
        return [constraints[0]] if constraints is not None else []
    if not isinstance(predicted, Rule) or not isinstance(gold, Rule):
        return []
    if len(predicted.premises) != len(gold.premises):
        return []
    candidates: set[tuple[tuple[str, str], ...]] = set()
    for gold_premises in itertools.permutations(gold.premises):
        variable_forward: dict[str, str] = {}
        variable_reverse: dict[str, str] = {}
        constant_pairs: list[tuple[str, str]] = []
        valid = True
        for left, right in zip(predicted.premises, gold_premises):
            aligned = _aligned_atom_constraints(
                left, right, variable_forward, variable_reverse
            )
            if aligned is None:
                valid = False
                break
            pairs, variable_forward, variable_reverse = aligned
            constant_pairs.extend(pairs)
        if not valid:
            continue
        conclusion = _aligned_atom_constraints(
            predicted.conclusion,
            gold.conclusion,
            variable_forward,
            variable_reverse,
        )
        if conclusion is None:
            continue
        constant_pairs.extend(conclusion[0])
        normalized = _normalize_constant_constraints(constant_pairs)
        if normalized is not None:
            candidates.add(normalized)
    return sorted(candidates)


def _aligned_atom_constraints(
    predicted: Atom,
    gold: Atom,
    variable_forward: Mapping[str, str],
    variable_reverse: Mapping[str, str],
) -> tuple[
    tuple[tuple[str, str], ...], dict[str, str], dict[str, str]
] | None:
    if (
        predicted.predicate != gold.predicate
        or predicted.negated != gold.negated
        or len(predicted.arguments) != len(gold.arguments)
    ):
        return None
    forward = dict(variable_forward)
    reverse = dict(variable_reverse)
    constants: list[tuple[str, str]] = []
    for left, right in zip(predicted.arguments, gold.arguments):
        if left.kind != right.kind:
            return None
        if left.kind == "constant":
            constants.append((left.name, right.name))
            continue
        existing = forward.get(left.name)
        reverse_existing = reverse.get(right.name)
        if (existing is not None and existing != right.name) or (
            reverse_existing is not None and reverse_existing != left.name
        ):
            return None
        forward[left.name] = right.name
        reverse[right.name] = left.name
    return tuple(constants), forward, reverse


def _normalize_constant_constraints(
    pairs: Iterable[tuple[str, str]],
) -> tuple[tuple[str, str], ...] | None:
    forward: dict[str, str] = {}
    reverse: dict[str, str] = {}
    for left, right in pairs:
        if (left in forward and forward[left] != right) or (
            right in reverse and reverse[right] != left
        ):
            return None
        forward[left] = right
        reverse[right] = left
    return tuple(sorted(forward.items()))


def _extend_constant_bijection(
    mapping: Mapping[str, str], candidate: Iterable[tuple[str, str]]
) -> dict[str, str] | None:
    forward = dict(mapping)
    reverse = {right: left for left, right in forward.items()}
    for left, right in candidate:
        if (left in forward and forward[left] != right) or (
            right in reverse and reverse[right] != left
        ):
            return None
        forward[left] = right
        reverse[right] = left
    return forward


def _find_complete_bijection(
    groups: Sequence[
        tuple[tuple[str, str], list[tuple[tuple[str, str], ...]]]
    ],
) -> dict[str, str] | None:
    def search(index: int, mapping: dict[str, str]) -> dict[str, str] | None:
        if index == len(groups):
            return mapping
        for candidate in groups[index][1]:
            extended = _extend_constant_bijection(mapping, candidate)
            if extended is None:
                continue
            result = search(index + 1, extended)
            if result is not None:
                return result
        return None

    return search(0, {})


def _greedy_consistent_bijection(
    groups: Sequence[
        tuple[tuple[str, str], list[tuple[tuple[str, str], ...]]]
    ],
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for _, candidates in groups:
        for candidate in candidates:
            extended = _extend_constant_bijection(mapping, candidate)
            if extended is not None:
                mapping = extended
                break
    return mapping


def _candidate_satisfied(
    candidate: Iterable[tuple[str, str]], mapping: Mapping[str, str]
) -> bool:
    return all(mapping.get(left) == right for left, right in candidate)


def _normalized_atom_key(
    atom: Atom, variable_map: Mapping[str, str] | None = None
) -> tuple[Any, ...]:
    mapping = variable_map or {}
    return (
        atom.predicate.lower(),
        tuple(
            (
                term.kind,
                mapping.get(term.name, term.name.lower())
                if term.kind == "variable"
                else term.name.lower(),
            )
            for term in atom.arguments
        ),
        atom.negated,
    )


def _normalized_rule_key(rule: Rule) -> tuple[Any, ...]:
    variables = sorted(
        set().union(
            *(atom.variables for atom in (*rule.premises, rule.conclusion))
        )
    )
    if not variables:
        return (
            frozenset(_normalized_atom_key(atom) for atom in rule.premises),
            _normalized_atom_key(rule.conclusion),
        )
    canonical_names = tuple(f"V{index}" for index in range(len(variables)))
    candidates: list[tuple[Any, ...]] = []
    for permutation in itertools.permutations(canonical_names):
        mapping = dict(zip(variables, permutation))
        candidates.append(
            (
                tuple(sorted(_normalized_atom_key(atom, mapping) for atom in rule.premises)),
                _normalized_atom_key(rule.conclusion, mapping),
            )
        )
    return min(candidates)


def _normalized_units(formalization: Formalization) -> set[tuple[Any, ...]]:
    result: set[tuple[Any, ...]] = set()
    for fact in formalization.facts:
        result.add(("fact", fact.source_id, _normalized_atom_key(fact)))
    for rule in formalization.rules:
        result.add(("rule", rule.source_id, _normalized_rule_key(rule)))
    result.add(
        (
            "query",
            formalization.query.source_id,
            _normalized_atom_key(formalization.query),
        )
    )
    return result
