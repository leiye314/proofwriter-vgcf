"""Static checks for constrained formalizations.

The validator catches machine-checkable structural problems. It cannot prove
semantic fidelity: a fluent but wrong predicate choice or a reversed relation
may remain structurally valid.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

from .data import Example, ModelExample
from .errors import SchemaError
from .ir_v2 import V2IR, normalize_singleton_rule_if, strip_optional_fence
from .schema import Atom, Formalization


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: str
    message: str
    path: str
    repair_hint: str
    severity: str = "hard"

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "message": self.message,
            "path": self.path,
            "repair_hint": self.repair_hint,
            "severity": self.severity,
        }


@dataclass(frozen=True, slots=True)
class ValidationResult:
    formalization: Formalization | None
    issues: tuple[ValidationIssue, ...]
    json_parse_valid: bool
    # ``schema_valid`` remains the operational alias used by the route and is
    # always equal to ``normalized_schema_valid`` for model JSON.
    schema_valid: bool
    strict_schema_valid: bool
    normalized_schema_valid: bool
    singleton_if_normalized_count: int
    strict_formalization: Formalization | None

    @property
    def valid(self) -> bool:
        """Backward-compatible alias for hard validity."""

        return self.hard_valid

    @property
    def hard_valid(self) -> bool:
        return self.formalization is not None and not any(
            issue.severity == "hard" for issue in self.issues
        )

    @property
    def hard_issues(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "hard")

    @property
    def soft_issues(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "soft")

    @property
    def trust_score(self) -> float:
        return 1.0 / (1.0 + len(self.soft_issues)) if self.hard_valid else 0.0


class StaticValidator:
    def validate_json(
        self, raw: str, example: Example | ModelExample
    ) -> ValidationResult:
        try:
            value: Any = json.loads(strip_optional_fence(raw))
        except json.JSONDecodeError as exc:
            return ValidationResult(
                None,
                (
                    ValidationIssue(
                        "invalid_json",
                        str(exc),
                        "$",
                        "Return one valid JSON object without prose or Markdown fences.",
                    ),
                ),
                False,
                False,
                False,
                False,
                0,
                None,
            )
        strict_formalization: Formalization | None = None
        try:
            strict_formalization = V2IR.from_dict(value).to_formalization()
        except SchemaError:
            pass
        normalized_value, normalized_count = normalize_singleton_rule_if(value)
        try:
            formalization = V2IR.from_dict(normalized_value).to_formalization()
        except SchemaError as exc:
            return ValidationResult(
                None,
                (
                    ValidationIssue(
                        "schema_error",
                        str(exc),
                        "$",
                        "Add required fields, remove unknown fields, and use the documented types.",
                    ),
                ),
                True,
                False,
                strict_formalization is not None,
                False,
                normalized_count,
                strict_formalization,
            )
        validated = self.validate(formalization, example)
        return ValidationResult(
            validated.formalization,
            validated.issues,
            True,
            True,
            strict_formalization is not None,
            True,
            normalized_count,
            strict_formalization,
        )

    def validate(
        self, formalization: Formalization, example: Example | ModelExample
    ) -> ValidationResult:
        issues: list[ValidationIssue] = []
        expected_sources = {sentence.source_id for sentence in example.theory}
        source_text = {sentence.source_id: sentence.text for sentence in example.theory}
        structures: list[tuple[str, str]] = [
            (fact.source_id or "", f"facts[{index}]")
            for index, fact in enumerate(formalization.facts)
        ] + [
            (rule.source_id, f"rules[{index}]")
            for index, rule in enumerate(formalization.rules)
        ]
        seen_sources: dict[str, str] = {}
        for source_id, path in structures:
            if not source_id:
                issues.append(
                    ValidationIssue(
                        "missing_source_id", "source_id is empty", path, "Set the source_id to its theory line."
                    )
                )
            elif source_id in seen_sources:
                issues.append(
                    ValidationIssue(
                        "duplicate_source_id",
                        f"{source_id} is used by both {seen_sources[source_id]} and {path}",
                        path,
                        "Represent each original theory sentence exactly once.",
                    )
                )
            else:
                seen_sources[source_id] = path
            if source_id and source_id not in expected_sources:
                issues.append(
                    ValidationIssue(
                        "unknown_source_id",
                        f"{source_id} is not an input theory source",
                        path,
                        "Use one of the generated s1..sN source IDs.",
                    )
                )
        for source_id in sorted(expected_sources - set(seen_sources)):
            issues.append(
                ValidationIssue(
                    "omitted_source",
                    f"original theory sentence {source_id} was omitted",
                    "$",
                    f"Add exactly one fact or rule with source_id {source_id}.",
                )
            )

        if formalization.query.source_id != example.question.source_id:
            issues.append(
                ValidationIssue(
                    "query_source_mismatch",
                    f"query source_id must be {example.question.source_id}",
                    "query.source_id",
                    f"Set query.source_id to {example.question.source_id}.",
                )
            )

        self._check_variable_locations(formalization, issues)
        self._check_entities(formalization, example, issues)
        self._check_arities(formalization, issues)
        self._check_rule_safety(formalization, issues)
        self._check_query(formalization, example, issues)
        self._check_query_not_fact(formalization, issues)
        self._check_duplicates(formalization, issues)
        self._check_negation(formalization, source_text, example.question.text, issues)
        return ValidationResult(
            formalization,
            tuple(issues),
            True,
            True,
            True,
            True,
            0,
            formalization,
        )

    @staticmethod
    def _check_variable_locations(
        formalization: Formalization, issues: list[ValidationIssue]
    ) -> None:
        for index, fact in enumerate(formalization.facts):
            if fact.variables:
                issues.append(
                    ValidationIssue(
                        "variable_in_fact",
                        f"facts must be ground; found variables {sorted(fact.variables)}",
                        f"facts[{index}].atom",
                        "Use lowercase constants in facts; variables are allowed only in rules.",
                    )
                )
        if formalization.query.variables:
            issues.append(
                ValidationIssue(
                    "non_ground_query",
                    f"query contains variables {sorted(formalization.query.variables)}",
                    "query.atom",
                    "Replace query variables with the exact lowercase question constants.",
                )
            )

    @staticmethod
    def _check_entities(
        formalization: Formalization,
        example: Example | ModelExample,
        issues: list[ValidationIssue],
    ) -> None:
        input_text = " ".join(
            [sentence.text for sentence in example.theory] + [example.question.text]
        )
        for atom_index, atom in enumerate(formalization.all_atoms()):
            for constant in atom.constants:
                alignment = _constant_alignment(constant, input_text)
                if alignment == "missing":
                    issues.append(
                        ValidationIssue(
                            "phantom_entity",
                            f"constant {constant!r} does not occur in the natural-language input",
                            f"atoms[{atom_index}]",
                            "Use only entities explicitly mentioned in theory or question.",
                            "soft",
                        )
                    )
                elif alignment == "article_omission":
                    issues.append(
                        ValidationIssue(
                            "inconsistent_entity_alias",
                            f"constant {constant!r} omits an article from a matching input phrase",
                            f"atoms[{atom_index}]",
                            "Treat this as advisory: verify the exact article-preserving constant manually.",
                            "soft",
                        )
                    )

    @staticmethod
    def _check_arities(
        formalization: Formalization, issues: list[ValidationIssue]
    ) -> None:
        arities: dict[str, set[int]] = {}
        for atom in formalization.all_atoms():
            arities.setdefault(atom.predicate, set()).add(len(atom.arguments))
        for predicate, values in sorted(arities.items()):
            if len(values) > 1:
                issues.append(
                    ValidationIssue(
                        "arity_mismatch",
                        f"predicate {predicate!r} has inconsistent arities {sorted(values)}",
                        "$",
                        "Use one argument count consistently for each predicate.",
                    )
                )

    @staticmethod
    def _check_rule_safety(
        formalization: Formalization, issues: list[ValidationIssue]
    ) -> None:
        for index, rule in enumerate(formalization.rules):
            body_variables = set().union(*(atom.variables for atom in rule.premises))
            unbound = rule.conclusion.variables - body_variables
            if unbound:
                issues.append(
                    ValidationIssue(
                        "unbound_conclusion_variable",
                        f"conclusion variables are not bound in premises: {sorted(unbound)}",
                        f"rules[{index}].conclusion",
                        "Every conclusion variable must also occur in a rule premise.",
                    )
                )

    @staticmethod
    def _stem(token: str) -> str:
        token = token.lower()
        if token.endswith("ies") and len(token) > 3:
            return token[:-3] + "y"
        if token.endswith("s") and len(token) > 2:
            return token[:-1]
        return token

    def _check_query(
        self,
        formalization: Formalization,
        example: Example | ModelExample,
        issues: list[ValidationIssue],
    ) -> None:
        question_tokens = set(re.findall(r"[a-z0-9_]+", example.question.text.lower()))
        stemmed = {self._stem(token) for token in question_tokens}
        predicate = self._stem(formalization.query.predicate)
        if predicate not in stemmed:
            issues.append(
                ValidationIssue(
                    "query_predicate_mismatch",
                    f"query predicate {formalization.query.predicate!r} is not aligned with the question",
                    "query.predicate",
                    "Formalize the predicate stated in the original question.",
                    "soft",
                )
            )
        for constant in formalization.query.constants:
            if _constant_alignment(constant, example.question.text) == "missing":
                issues.append(
                    ValidationIssue(
                        "query_entity_mismatch",
                        f"query entity {constant!r} is absent from the question",
                        "query.arguments",
                        "Use only question entities and preserve their roles/order.",
                        "soft",
                    )
                )

    @staticmethod
    def _check_query_not_fact(
        formalization: Formalization, issues: list[ValidationIssue]
    ) -> None:
        query_key = formalization.query.structural_key()
        for index, fact in enumerate(formalization.facts):
            if fact.structural_key() == query_key and fact.source_id == "q1":
                issues.append(
                    ValidationIssue(
                        "query_added_as_fact",
                        "the query was copied into the fact set using q1",
                        f"facts[{index}]",
                        "Remove the q1 fact; the question is not evidence.",
                    )
                )

    @staticmethod
    def _check_duplicates(
        formalization: Formalization, issues: list[ValidationIssue]
    ) -> None:
        fact_keys: dict[tuple[Any, ...], int] = {}
        for index, fact in enumerate(formalization.facts):
            key = fact.structural_key()
            if key in fact_keys:
                issues.append(
                    ValidationIssue(
                        "duplicate_fact",
                        f"fact duplicates facts[{fact_keys[key]}]",
                        f"facts[{index}]",
                        "Keep only one copy and preserve the correct source mapping.",
                        "soft",
                    )
                )
            else:
                fact_keys[key] = index
        rule_keys: dict[tuple[Any, ...], int] = {}
        for index, rule in enumerate(formalization.rules):
            key = rule.structural_key()
            if key in rule_keys:
                issues.append(
                    ValidationIssue(
                        "duplicate_rule",
                        f"rule duplicates rules[{rule_keys[key]}]",
                        f"rules[{index}]",
                        "Keep only one copy and preserve the correct source mapping.",
                        "soft",
                    )
                )
            else:
                rule_keys[key] = index

    @staticmethod
    def _has_negation(text: str) -> bool:
        return bool(re.search(r"\b(?:not|no|never)\b", text, flags=re.IGNORECASE))

    def _check_negation(
        self,
        formalization: Formalization,
        source_text: dict[str, str],
        question_text: str,
        issues: list[ValidationIssue],
    ) -> None:
        for index, fact in enumerate(formalization.facts):
            text = source_text.get(fact.source_id or "")
            if text is not None and self._has_negation(text) != fact.negated:
                issues.append(
                    ValidationIssue(
                        "fact_negation_mismatch",
                        "fact negation disagrees with an explicit natural-language negation marker",
                        f"facts[{index}].negated",
                        "Set negated to match the source sentence's explicit negation.",
                        "soft",
                    )
                )
        for index, rule in enumerate(formalization.rules):
            text = source_text.get(rule.source_id)
            has_structured_negation = any(
                atom.negated for atom in (*rule.premises, rule.conclusion)
            )
            if text is not None and self._has_negation(text) != has_structured_negation:
                issues.append(
                    ValidationIssue(
                        "rule_negation_mismatch",
                        "rule negation disagrees with an explicit natural-language negation marker",
                        f"rules[{index}]",
                        "Preserve explicit negation in the corresponding premise or conclusion.",
                        "soft",
                    )
                )
        if self._has_negation(question_text) != formalization.query.negated:
            issues.append(
                ValidationIssue(
                    "query_negation_mismatch",
                    "query negation disagrees with the question",
                    "query.negated",
                    "Set query.negated to match the original question.",
                    "soft",
                )
            )


def issue_dicts(issues: Iterable[ValidationIssue]) -> list[dict[str, str]]:
    return [issue.to_dict() for issue in issues]


def _constant_alignment(constant: str, text: str) -> str:
    """Return a conservative lexical signal; it is never a hard decision."""

    constant_tokens = constant.lower().split("_")
    text_tokens = re.findall(r"[a-z0-9]+", text.lower())
    articles = {"a", "an", "the"}
    for start in range(len(text_tokens) - len(constant_tokens) + 1):
        if text_tokens[start : start + len(constant_tokens)] != constant_tokens:
            continue
        if (
            constant_tokens[0] not in articles
            and start > 0
            and text_tokens[start - 1] in articles
        ):
            return "article_omission"
        return "exact"
    return "missing"


def merge_repair_issues(
    issues: Iterable[ValidationIssue], limit: int = 5, *, severity: str = "hard"
) -> list[dict[str, Any]]:
    """Merge repeated root causes into a bounded severity-specific list."""

    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for issue in issues:
        if issue.severity != severity:
            continue
        key = (issue.severity, issue.code)
        if key not in grouped:
            grouped[key] = {
                "code": issue.code,
                "severity": issue.severity,
                "message": issue.message,
                "paths": [],
                "repair_hint": issue.repair_hint,
                "count": 0,
            }
        item = grouped[key]
        item["count"] += 1
        if issue.path not in item["paths"]:
            item["paths"].append(issue.path)
    ordered = sorted(grouped.values(), key=lambda item: item["code"])
    return ordered[:limit]
