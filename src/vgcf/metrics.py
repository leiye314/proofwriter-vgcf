"""Transparent classification and operational metrics."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence

LABELS = ("True", "False", "Unknown")


def classification_metrics(
    gold: Sequence[str], predicted: Sequence[str]
) -> dict[str, Any]:
    if len(gold) != len(predicted):
        raise ValueError("gold and predicted lengths differ")
    if not gold:
        raise ValueError("metrics require at least one example")
    correct = sum(expected == actual for expected, actual in zip(gold, predicted))
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for label in LABELS:
        true_positive = sum(g == label and p == label for g, p in zip(gold, predicted))
        false_positive = sum(g != label and p == label for g, p in zip(gold, predicted))
        false_negative = sum(g == label and p != label for g, p in zip(gold, predicted))
        support = sum(g == label for g in gold)
        precision = _zero_when_undefined(
            true_positive, true_positive + false_positive
        )
        recall = _zero_when_undefined(
            true_positive, true_positive + false_negative
        )
        f1 = _zero_when_undefined(2 * precision * recall, precision + recall)
        f1_values.append(f1)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    errors = sum(value not in LABELS for value in predicted)
    return {
        "count": len(gold),
        "accuracy": correct / len(gold),
        "macro_f1": sum(f1_values) / len(f1_values),
        "error_rate": errors / len(gold),
        "per_class": per_class,
        "confusion": _confusion(gold, predicted),
    }


def aggregate_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["method"])].append(record)
    result: dict[str, Any] = {}
    for method, rows in sorted(grouped.items()):
        metrics = classification_metrics(
            [str(row["gold_label"]) for row in rows],
            [str(row["predicted_label"]) for row in rows],
        )
        formalization_method = method in {
            "plain",
            "constrained",
            "gate_cot",
            "repair_only",
            "vgcf2",
            "vgcf",
        }
        json_valid = sum(row.get("json_parse_valid") is True for row in rows)
        strict_schema_valid = sum(
            _metric_bool(row, "strict_schema_valid", "schema_valid") for row in rows
        )
        normalized_schema_valid = sum(
            _metric_bool(row, "normalized_schema_valid", "schema_valid")
            for row in rows
        )
        schema_valid = normalized_schema_valid
        hard_valid = sum(row.get("hard_validator_pass") is True for row in rows)
        strict_solver_executable = sum(
            _metric_bool(row, "strict_solver_executable", "solver_executable")
            for row in rows
        )
        normalized_solver_executable = sum(
            _metric_bool(
                row, "normalized_solver_executable", "solver_executable"
            )
            for row in rows
        )
        solver_executable = normalized_solver_executable
        repair_attempts = sum(bool(row.get("repair_attempted")) for row in rows)
        repair_improved = sum(bool(row.get("repair_improved")) for row in rows)
        repair_regressions = sum(bool(row.get("repair_regression")) for row in rows)
        repair_regression_layers = Counter(
            str(layer)
            for row in rows
            for layer in (row.get("repair_regression_layers") or [])
        )
        covered = [row for row in rows if bool(row.get("coverage"))]
        correct_count = sum(bool(row.get("final_label_correct")) for row in rows)
        route_metrics = _group_accuracy(rows, "route_source")
        executable_gate_blocks = [
            row
            for row in rows
            if row.get("route_source") == "cot_fallback"
            and row.get("solver_executable") is True
        ]
        false_blocks = sum(
            row.get("shadow_solver_label") == row.get("gold_label")
            for row in executable_gate_blocks
        )
        semantic_rows = [
            row for row in rows if isinstance(row.get("formalization_semantics"), Mapping)
        ]
        metrics.update(
            {
                "repair_rate": repair_attempts / len(rows),
                "repair_attempt_count": repair_attempts,
                "repair_improved_count": repair_improved,
                "repair_improved_rate": _undefined_if_empty(
                    repair_improved, repair_attempts
                ),
                "repair_regression_count": repair_regressions,
                "repair_regression_rate": _undefined_if_empty(
                    repair_regressions, repair_attempts
                ),
                "repair_regression_by_layer": dict(
                    sorted(repair_regression_layers.items())
                ),
                "validator_trigger_count": repair_attempts,
                "validation_issue_count": sum(
                    len(row.get("validation_errors") or []) for row in rows
                ),
                "json_parse_valid_count": json_valid
                if formalization_method
                else None,
                "json_parse_valid_rate": json_valid / len(rows)
                if formalization_method
                else None,
                "schema_valid_count": schema_valid if formalization_method else None,
                "schema_valid_rate": schema_valid / len(rows)
                if formalization_method
                else None,
                "strict_schema_valid_count": strict_schema_valid
                if formalization_method
                else None,
                "strict_schema_valid_rate": strict_schema_valid / len(rows)
                if formalization_method
                else None,
                "normalized_schema_valid_count": normalized_schema_valid
                if formalization_method
                else None,
                "normalized_schema_valid_rate": normalized_schema_valid / len(rows)
                if formalization_method
                else None,
                "hard_validator_pass_count": hard_valid if formalization_method else None,
                "hard_validator_pass_rate": hard_valid / len(rows)
                if formalization_method
                else None,
                "soft_issue_count": sum(int(row.get("soft_issue_count") or 0) for row in rows)
                if formalization_method
                else None,
                "mean_soft_issue_count": sum(
                    int(row.get("soft_issue_count") or 0) for row in rows
                )
                / len(rows)
                if formalization_method
                else None,
                "solver_executable_count": solver_executable
                if formalization_method
                else None,
                "strict_solver_executable_count": strict_solver_executable
                if formalization_method
                else None,
                "strict_solver_executable_rate": strict_solver_executable
                / len(rows)
                if formalization_method
                else None,
                "normalized_solver_executable_count": (
                    normalized_solver_executable if formalization_method else None
                ),
                "normalized_solver_executable_rate": (
                    normalized_solver_executable / len(rows)
                    if formalization_method
                    else None
                ),
                "singleton_if_normalized_program_count": sum(
                    int(row.get("singleton_if_normalized_count") or 0) > 0
                    for row in rows
                )
                if formalization_method
                else None,
                "singleton_if_normalized_rule_count": sum(
                    int(row.get("singleton_if_normalized_count") or 0)
                    for row in rows
                )
                if formalization_method
                else None,
                "solver_executable_rate": solver_executable / len(rows)
                if formalization_method
                else None,
                "shadow_solver_executable_count": solver_executable
                if formalization_method
                else None,
                "shadow_solver_executable_rate": solver_executable / len(rows)
                if formalization_method
                else None,
                "gold_semantic_match_count": sum(
                    row.get("gold_semantic_match") is True for row in rows
                )
                if semantic_rows
                else None,
                "gold_semantic_match_rate": _undefined_if_empty(
                    sum(row.get("gold_semantic_match") is True for row in rows),
                    len(semantic_rows),
                )
                if semantic_rows
                else None,
                "gold_semantic_exact_lower_bound_rate": _undefined_if_empty(
                    sum(row.get("gold_semantic_match") is True for row in rows),
                    len(semantic_rows),
                )
                if semantic_rows
                else None,
                "gold_semantic_normalized_diagnostic_match_count": sum(
                    row.get("gold_semantic_normalized_diagnostic_match") is True
                    for row in rows
                )
                if semantic_rows
                else None,
                "gold_semantic_normalized_diagnostic_match_rate": _undefined_if_empty(
                    sum(
                        row.get("gold_semantic_normalized_diagnostic_match") is True
                        for row in rows
                    ),
                    len(semantic_rows),
                )
                if semantic_rows
                else None,
                "gold_semantic_renaming_invariant_match_count": sum(
                    row.get("gold_semantic_renaming_invariant_match") is True
                    for row in rows
                )
                if semantic_rows
                else None,
                "gold_semantic_renaming_invariant_match_rate": (
                    _undefined_if_empty(
                        sum(
                            row.get("gold_semantic_renaming_invariant_match")
                            is True
                            for row in rows
                        ),
                        len(semantic_rows),
                    )
                    if semantic_rows
                    else None
                ),
                "mean_formalization_precision": _undefined_if_empty(
                    sum(
                        float(row["formalization_semantics"].get("precision", 0))
                        for row in semantic_rows
                    ),
                    len(semantic_rows),
                )
                if semantic_rows
                else None,
                "mean_formalization_recall": _undefined_if_empty(
                    sum(
                        float(row["formalization_semantics"].get("recall", 0))
                        for row in semantic_rows
                    ),
                    len(semantic_rows),
                )
                if semantic_rows
                else None,
                "mean_strict_semantic_precision": _semantic_mean(
                    semantic_rows, "strict_semantic_precision"
                ),
                "mean_strict_semantic_recall": _semantic_mean(
                    semantic_rows, "strict_semantic_recall"
                ),
                "mean_renaming_invariant_semantic_precision": _semantic_mean(
                    semantic_rows, "renaming_invariant_semantic_precision"
                ),
                "mean_renaming_invariant_semantic_recall": _semantic_mean(
                    semantic_rows, "renaming_invariant_semantic_recall"
                ),
                "final_label_correct_count": correct_count,
                "accuracy_wilson_95": wilson_interval(correct_count, len(rows)),
                "route_source_counts": dict(
                    Counter(str(row.get("route_source")) for row in rows)
                ),
                "route_source": dict(
                    Counter(str(row.get("route_source")) for row in rows)
                ),
                "accuracy_by_route_source": route_metrics,
                "solver_initial_count": sum(
                    row.get("route_source") == "solver_initial" for row in rows
                ),
                "solver_repair_count": sum(
                    row.get("route_source") == "solver_repair" for row in rows
                ),
                "cot_fallback_count": sum(
                    row.get("route_source") == "cot_fallback" for row in rows
                ),
                "answer_from_solver_count": sum(
                    bool(row.get("answer_from_solver")) for row in rows
                ),
                "answer_from_solver_rate": _undefined_if_empty(
                    sum(bool(row.get("answer_from_solver")) for row in rows),
                    len(rows),
                ),
                "coverage": len(covered) / len(rows),
                "coverage_wilson_95": wilson_interval(len(covered), len(rows)),
                "selective_accuracy": _undefined_if_empty(
                    sum(bool(row.get("final_label_correct")) for row in covered),
                    len(covered),
                ),
                "selective_accuracy_wilson_95": wilson_interval(
                    sum(bool(row.get("final_label_correct")) for row in covered),
                    len(covered),
                ),
                "fallback_rate": sum(bool(row.get("fallback_used")) for row in rows)
                / len(rows),
                "gate_block_count": sum(
                    row.get("route_source") == "cot_fallback" for row in rows
                ),
                "gate_executable_shadow_count": len(executable_gate_blocks),
                "gate_false_block_count": false_blocks,
                "gate_false_block_rate": _undefined_if_empty(
                    false_blocks, len(executable_gate_blocks)
                ),
                "infrastructure_error_count": sum(
                    bool(row.get("infrastructure_error")) for row in rows
                ),
                "method_error_count": sum(
                    bool(row.get("method_error")) for row in rows
                ),
                "method_attributed_call_count": sum(
                    int(row.get("call_count", 0)) for row in rows
                ),
                "total_call_count": sum(
                    int(row.get("call_count", 0)) for row in rows
                ),
                "method_row_attributed_latency_ms_total": sum(
                    float(row.get("latency_ms", 0)) for row in rows
                ),
                "method_row_attributed_latency_ms_mean": sum(
                    float(row.get("latency_ms", 0)) for row in rows
                )
                / len(rows),
                "method_row_latency_is_physical": False,
                "prompt_tokens_attributed": sum(
                    int(row.get("tokens", {}).get("prompt", 0)) for row in rows
                ),
                "completion_tokens_attributed": sum(
                    int(row.get("tokens", {}).get("completion", 0)) for row in rows
                ),
                "total_tokens_attributed": sum(
                    int(row.get("tokens", {}).get("total", 0)) for row in rows
                ),
                "total_cost_usd": sum(float(row.get("cost_usd", 0)) for row in rows),
                "cache_hit_rate": sum(bool(row.get("cached")) for row in rows) / len(rows),
                "gold_distribution": dict(Counter(str(row["gold_label"]) for row in rows)),
                "by_label": _group_accuracy(rows, "gold_label"),
                "by_depth": _depth_accuracy(rows),
                "within_label_depth": _within_label_depth(rows),
                "cot_refine_contract_valid_count": sum(
                    row.get("cot_refine_contract_valid") is True for row in rows
                )
                if method == "cot_refine"
                else None,
                "cot_refine_contract_failure_count": sum(
                    row.get("cot_refine_contract_valid") is False for row in rows
                )
                if method == "cot_refine"
                else None,
                "cot_refine_contract_valid_rate": (
                    sum(row.get("cot_refine_contract_valid") is True for row in rows)
                    / len(rows)
                )
                if method == "cot_refine"
                else None,
            }
        )
        result[method] = metrics
    return result


def _zero_when_undefined(numerator: float, denominator: float) -> float:
    """Use the conventional zero value for classification component metrics."""

    return numerator / denominator if denominator else 0.0


def _undefined_if_empty(
    numerator: float, denominator: float
) -> float | None:
    """Return ``None`` when an empty conditional subset makes a rate unknowable."""

    return numerator / denominator if denominator else None


def _metric_bool(
    row: Mapping[str, Any], preferred: str, legacy: str
) -> bool:
    value = row.get(preferred) if preferred in row else row.get(legacy)
    return value is True


def _semantic_mean(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    values = [
        float(row["formalization_semantics"][field])
        for row in rows
        if field in row["formalization_semantics"]
    ]
    return sum(values) / len(values) if values else None


def wilson_interval(
    successes: int, count: int, z: float = 1.959963984540054
) -> dict[str, float] | None:
    """Dependency-free two-sided Wilson score interval."""

    if count <= 0:
        return None
    proportion = successes / count
    denominator = 1.0 + z * z / count
    center = (proportion + z * z / (2.0 * count)) / denominator
    margin = (
        z
        * ((proportion * (1.0 - proportion) / count + z * z / (4.0 * count * count)) ** 0.5)
        / denominator
    )
    return {"low": max(0.0, center - margin), "high": min(1.0, center + margin)}


def _confusion(gold: Sequence[str], predicted: Sequence[str]) -> dict[str, dict[str, int]]:
    columns = (*LABELS, "Error")
    matrix = {expected: {actual: 0 for actual in columns} for expected in LABELS}
    for expected, actual in zip(gold, predicted):
        actual_column = actual if actual in LABELS else "Error"
        if expected in matrix:
            matrix[expected][actual_column] += 1
    return matrix


def _group_accuracy(
    rows: Sequence[Mapping[str, Any]], field: str
) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(field))].append(row)
    return {
        key: {
            "count": len(group),
            "correct": sum(
                str(row.get("gold_label")) == str(row.get("predicted_label"))
                for row in group
            ),
            "accuracy": sum(
                str(row.get("gold_label")) == str(row.get("predicted_label"))
                for row in group
            )
            / len(group),
        }
        for key, group in sorted(grouped.items())
    }


def _depth_accuracy(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("depth"))].append(row)
    result: dict[str, dict[str, Any]] = {}
    for depth, group in sorted(grouped.items(), key=lambda item: _depth_sort_key(item[0])):
        correct = sum(
            str(row.get("gold_label")) == str(row.get("predicted_label"))
            for row in group
        )
        result[depth] = {
            "count": len(group),
            "correct": correct,
            "accuracy": correct / len(group),
            "label_composition": {
                label: sum(str(row.get("gold_label")) == label for row in group)
                for label in LABELS
            },
            "singleton_support": len(group) == 1,
        }
    return result


def _within_label_depth(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, float | int | None]]]:
    depths = sorted(
        {str(row.get("depth")) for row in rows}, key=_depth_sort_key
    )
    result: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for label in LABELS:
        by_depth: dict[str, dict[str, float | int | None]] = {}
        for depth in depths:
            group = [
                row
                for row in rows
                if str(row.get("gold_label")) == label
                and str(row.get("depth")) == depth
            ]
            correct = sum(
                str(row.get("predicted_label")) == label for row in group
            )
            by_depth[depth] = {
                "count": len(group),
                "correct": correct,
                "accuracy": _undefined_if_empty(correct, len(group)),
            }
        result[label] = by_depth
    return result


def _depth_sort_key(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)
