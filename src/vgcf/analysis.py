"""Offline result analysis; never calls a model."""

from __future__ import annotations

import csv
import html
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

from .metrics import aggregate_records
from .physical import (
    reconstruct_physical_accounting,
    render_physical_accounting_markdown,
)


def load_results(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"result line {line_number} is not an object")
            records.append(value)
    if not records:
        raise ValueError("result file contains no records")
    return records


def analyze_results(result_path: str | Path, output_directory: str | Path) -> dict[str, Any]:
    records = load_results(result_path)
    methods = aggregate_records(records)
    physical_accounting = (
        reconstruct_physical_accounting(records)
        if any(record.get("raw_output_paths") for record in records)
        else None
    )
    summary = {
        "result_path": str(Path(result_path).resolve()),
        "record_count": len(records),
        "example_count": len({str(record["example_id"]) for record in records}),
        "methods": methods,
        "operational_totals": _operational_totals(methods),
        "paired_mcnemar": paired_mcnemar(records),
        "physical_accounting": physical_accounting,
        "depth_analysis_policy": {
            "descriptive_only": True,
            "label_composition_required": True,
            "within_label_required": True,
            "singleton_depth_trend_claim_suppressed": True,
        },
    }
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    _write_csv(output / "method_metrics.csv", summary["methods"])
    _write_method_markdown(output / "method_metrics.md", summary["methods"])
    _write_depth_csvs(output, records, summary["methods"])
    (output / "depth_analysis.md").write_text(
        _render_depth_analysis(summary["methods"]), encoding="utf-8"
    )
    if physical_accounting is not None:
        (output / "physical_accounting.json").write_text(
            json.dumps(
                physical_accounting,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        (output / "physical_accounting.md").write_text(
            render_physical_accounting_markdown(physical_accounting),
            encoding="utf-8",
        )
    _write_svg(output / "accuracy.svg", summary["methods"])
    case_analyses = _case_analyses(records, limit=5)
    summary["case_analyses"] = case_analyses
    (output / "case_analyses.md").write_text(
        _render_case_analyses(case_analyses), encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def paired_mcnemar(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return paired correctness cells and an exact two-sided McNemar test."""

    by_method: dict[str, dict[str, bool]] = {}
    for row in records:
        method = str(row["method"])
        example_id = str(row["example_id"])
        correct = str(row.get("gold_label")) == str(row.get("predicted_label"))
        if example_id in by_method.setdefault(method, {}):
            raise ValueError(f"duplicate result for method={method} example={example_id}")
        by_method[method][example_id] = correct
    output: dict[str, Any] = {}
    for left, right in combinations(sorted(by_method), 2):
        shared = sorted(set(by_method[left]) & set(by_method[right]))
        both_correct = sum(
            by_method[left][item] and by_method[right][item] for item in shared
        )
        left_only = sum(
            by_method[left][item] and not by_method[right][item] for item in shared
        )
        right_only = sum(
            not by_method[left][item] and by_method[right][item] for item in shared
        )
        both_wrong = len(shared) - both_correct - left_only - right_only
        discordant = left_only + right_only
        if discordant:
            tail = sum(
                math.comb(discordant, value)
                for value in range(min(left_only, right_only) + 1)
            ) / (2**discordant)
            p_value = min(1.0, 2.0 * tail)
        else:
            p_value = 1.0
        output[f"{left}__vs__{right}"] = {
            "method_a": left,
            "method_b": right,
            "paired_count": len(shared),
            "both_correct": both_correct,
            "a_correct_b_wrong": left_only,
            "a_wrong_b_correct": right_only,
            "both_wrong": both_wrong,
            "discordant_count": discordant,
            "exact_two_sided_p": p_value,
        }
    return output


def _write_csv(path: Path, methods: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "count",
                "accuracy",
                "macro_f1",
                "error_rate",
                "repair_rate",
                "repair_improved_rate",
                "repair_regression_rate",
                "json_parse_valid_rate",
                "strict_schema_valid_rate",
                "normalized_schema_valid_rate",
                "schema_valid_rate",
                "hard_validator_pass_rate",
                "mean_soft_issue_count",
                "solver_executable_rate",
                "strict_solver_executable_rate",
                "normalized_solver_executable_rate",
                "answer_from_solver_rate",
                "gold_semantic_match_rate",
                "gold_semantic_exact_lower_bound_rate",
                "gold_semantic_normalized_diagnostic_match_rate",
                "gold_semantic_renaming_invariant_match_rate",
                "mean_formalization_precision",
                "mean_formalization_recall",
                "mean_strict_semantic_precision",
                "mean_strict_semantic_recall",
                "mean_renaming_invariant_semantic_precision",
                "mean_renaming_invariant_semantic_recall",
                "coverage",
                "selective_accuracy",
                "fallback_rate",
                "gate_false_block_rate",
                "cot_refine_contract_valid_rate",
                "infrastructure_error_count",
                "method_error_count",
                "validator_trigger_count",
                "repair_attempt_count",
                "repair_improved_count",
                "repair_regression_count",
                "method_attributed_call_count",
                "method_row_attributed_latency_ms_mean",
                "prompt_tokens_attributed",
                "completion_tokens_attributed",
                "total_tokens_attributed",
                "total_cost_usd",
                "cache_hit_rate",
            ],
        )
        writer.writeheader()
        for method, metrics in methods.items():
            writer.writerow(
                {
                    "method": method,
                    **{key: metrics.get(key) for key in writer.fieldnames[1:]},
                }
            )


def _write_method_markdown(path: Path, methods: dict[str, Any]) -> None:
    lines = [
        "# Method metrics",
        "",
        "Undefined conditional rates are rendered as N/A, never as 0%.",
        "",
        "| Method | Accuracy | Coverage | Selective accuracy | Gate false-block | CoT-Refine contract |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, metrics in methods.items():
        lines.append(
            f"| {method} | {format_optional_rate(metrics.get('accuracy'))} | "
            f"{format_optional_rate(metrics.get('coverage'))} | "
            f"{format_optional_rate(metrics.get('selective_accuracy'))} | "
            f"{format_optional_rate(metrics.get('gate_false_block_rate'))} | "
            f"{format_optional_rate(metrics.get('cot_refine_contract_valid_rate'))} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_depth_csvs(
    output: Path, records: list[dict[str, Any]], methods: dict[str, Any]
) -> None:
    del records  # Composition is already embedded in each method's deterministic metrics.
    with (output / "depth_by_method.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = [
            "method",
            "depth",
            "count",
            "correct",
            "accuracy",
            "True_count",
            "False_count",
            "Unknown_count",
            "singleton_support",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method, metrics in methods.items():
            for depth, item in metrics["by_depth"].items():
                composition = item["label_composition"]
                writer.writerow(
                    {
                        "method": method,
                        "depth": depth,
                        "count": item["count"],
                        "correct": item["correct"],
                        "accuracy": item["accuracy"],
                        "True_count": composition["True"],
                        "False_count": composition["False"],
                        "Unknown_count": composition["Unknown"],
                        "singleton_support": item["singleton_support"],
                    }
                )
    with (output / "within_label_depth.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = ["method", "label", "depth", "count", "correct", "accuracy"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method, metrics in methods.items():
            for label, by_depth in metrics["within_label_depth"].items():
                for depth, item in by_depth.items():
                    writer.writerow(
                        {
                            "method": method,
                            "label": label,
                            "depth": depth,
                            **item,
                        }
                    )


def _render_depth_analysis(methods: dict[str, Any]) -> str:
    lines = [
        "# Depth × label descriptive analysis",
        "",
        "所有总体 depth 行同时显示标签组成；within-label 空支持写 N/A。"
        "本报告不自动生成‘深度越大准确率越低’之类趋势结论，"
        "singleton depth 只作记录。",
        "",
    ]
    for method, metrics in methods.items():
        lines.extend(
            [
                f"## {method}",
                "",
                "| Depth | n | True | False | Unknown | Accuracy | Note |",
                "|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for depth, item in metrics["by_depth"].items():
            composition = item["label_composition"]
            note = "singleton; no trend inference" if item["singleton_support"] else ""
            lines.append(
                f"| {depth} | {item['count']} | {composition['True']} | "
                f"{composition['False']} | {composition['Unknown']} | "
                f"{format_optional_rate(item['accuracy'])} | {note} |"
            )
        lines.extend(
            [
                "",
                "| Label | Depth | n | Accuracy |",
                "|---|---:|---:|---:|",
            ]
        )
        for label, by_depth in metrics["within_label_depth"].items():
            for depth, item in by_depth.items():
                lines.append(
                    f"| {label} | {depth} | {item['count']} | "
                    f"{format_optional_rate(item['accuracy'])} |"
                )
        lines.append("")
    return "\n".join(lines) + "\n"


def format_optional_rate(value: Any) -> str:
    """Markdown representation for a possibly undefined rate."""

    if value is None:
        return "N/A"
    return f"{float(value):.1%}"


def _write_svg(path: Path, methods: dict[str, Any]) -> None:
    names = list(methods)
    width, height = 760, 420
    left, top, chart_height = 90, 45, 290
    bar_width = max(40, int(560 / max(1, len(names))))
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="380" y="25" text-anchor="middle" font-family="sans-serif" font-size="18">Smoke accuracy by method</text>',
    ]
    for tick in range(0, 11, 2):
        value = tick / 10
        y = top + chart_height * (1 - value)
        elements.append(f'<line x1="{left}" y1="{y:.1f}" x2="700" y2="{y:.1f}" stroke="#dddddd"/>')
        elements.append(
            f'<text x="78" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{value:.1f}</text>'
        )
    palette = ["#3b82f6", "#8b5cf6", "#14b8a6", "#f59e0b", "#ef4444"]
    for index, name in enumerate(names):
        accuracy = float(methods[name]["accuracy"])
        x = left + 28 + index * bar_width
        bar_height = accuracy * chart_height
        y = top + chart_height - bar_height
        elements.extend(
            [
                f'<rect x="{x}" y="{y:.1f}" width="{bar_width - 20}" height="{bar_height:.1f}" fill="{palette[index % len(palette)]}"/>',
                f'<text x="{x + (bar_width - 20) / 2:.1f}" y="{y - 7:.1f}" text-anchor="middle" font-family="sans-serif" font-size="12">{accuracy:.3f}</text>',
                f'<text x="{x + (bar_width - 20) / 2:.1f}" y="360" text-anchor="middle" font-family="sans-serif" font-size="12">{html.escape(name)}</text>',
            ]
        )
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def _operational_totals(methods: dict[str, Any]) -> dict[str, Any]:
    formal = [
        metrics
        for name, metrics in methods.items()
        if name in {"plain", "constrained", "gate_cot", "repair_only", "vgcf2", "vgcf"}
    ]
    formal_count = sum(int(metrics["count"]) for metrics in formal)
    json_valid = sum(int(metrics["json_parse_valid_count"] or 0) for metrics in formal)
    strict_schema_valid = sum(
        int(metrics["strict_schema_valid_count"] or 0) for metrics in formal
    )
    normalized_schema_valid = sum(
        int(metrics["normalized_schema_valid_count"] or 0) for metrics in formal
    )
    schema_valid = sum(int(metrics["schema_valid_count"] or 0) for metrics in formal)
    hard_valid = sum(int(metrics["hard_validator_pass_count"] or 0) for metrics in formal)
    solver_executable = sum(
        int(metrics["solver_executable_count"] or 0) for metrics in formal
    )
    strict_solver_executable = sum(
        int(metrics["strict_solver_executable_count"] or 0) for metrics in formal
    )
    normalized_solver_executable = sum(
        int(metrics["normalized_solver_executable_count"] or 0)
        for metrics in formal
    )
    return {
        "formalization_record_count": formal_count,
        "json_parse_valid_count": json_valid,
        "json_parse_valid_rate": json_valid / formal_count if formal_count else None,
        "strict_schema_valid_count": strict_schema_valid,
        "strict_schema_valid_rate": strict_schema_valid / formal_count
        if formal_count
        else None,
        "normalized_schema_valid_count": normalized_schema_valid,
        "normalized_schema_valid_rate": normalized_schema_valid / formal_count
        if formal_count
        else None,
        "schema_valid_count": schema_valid,
        "schema_valid_rate": schema_valid / formal_count if formal_count else None,
        "hard_validator_pass_count": hard_valid,
        "hard_validator_pass_rate": hard_valid / formal_count
        if formal_count
        else None,
        "strict_solver_executable_count": strict_solver_executable,
        "strict_solver_executable_rate": strict_solver_executable / formal_count
        if formal_count
        else None,
        "normalized_solver_executable_count": normalized_solver_executable,
        "normalized_solver_executable_rate": normalized_solver_executable
        / formal_count
        if formal_count
        else None,
        "solver_executable_count": solver_executable,
        "solver_executable_rate": solver_executable / formal_count
        if formal_count
        else None,
        "validator_trigger_count": sum(
            int(metrics["validator_trigger_count"]) for metrics in methods.values()
        ),
        "repair_attempt_count": sum(
            int(metrics["repair_attempt_count"]) for metrics in methods.values()
        ),
        "repair_improved_count": sum(
            int(metrics["repair_improved_count"]) for metrics in methods.values()
        ),
        "repair_regression_count": sum(
            int(metrics["repair_regression_count"]) for metrics in methods.values()
        ),
        "infrastructure_error_count": sum(
            int(metrics["infrastructure_error_count"]) for metrics in methods.values()
        ),
        "method_error_count": sum(
            int(metrics["method_error_count"]) for metrics in methods.values()
        ),
        "explicit_error_count": sum(
            round(float(metrics["error_rate"]) * int(metrics["count"]))
            for metrics in methods.values()
        ),
    }


def _case_analyses(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["example_id"]), []).append(record)

    def priority(rows: list[dict[str, Any]]) -> tuple[int, int, str]:
        predictions = {str(row.get("predicted_label")) for row in rows}
        errors = sum(bool(row.get("error")) for row in rows)
        repairs = sum(bool(row.get("repair_triggered")) for row in rows)
        disagreements = len(predictions) - 1
        return (errors * 100 + repairs * 10 + disagreements, disagreements, str(rows[0]["example_id"]))

    chosen = sorted(grouped.values(), key=priority, reverse=True)[:limit]
    analyses: list[dict[str, Any]] = []
    for rows in chosen:
        rows = sorted(rows, key=lambda row: str(row["method"]))
        predictions = {str(row["method"]): str(row["predicted_label"]) for row in rows}
        gold = str(rows[0]["gold_label"])
        errors = {
            str(row["method"]): str(row["error"])
            for row in rows
            if row.get("error")
        }
        repair_methods = [
            str(row["method"]) for row in rows if row.get("repair_triggered")
        ]
        unique_predictions = set(predictions.values())
        if errors:
            observation = "At least one method failed explicitly; inspect its saved raw response and validator issues."
        elif len(unique_predictions) > 1:
            observation = "Methods disagree on the same whitelisted input, indicating a method-sensitive reasoning or formalization case."
        elif next(iter(unique_predictions)) == gold:
            observation = "All recorded methods agree with the evaluator-only label on this case."
        else:
            observation = "All recorded methods agree with one another but not with the evaluator-only label."
        analyses.append(
            {
                "example_id": str(rows[0]["example_id"]),
                "depth": rows[0].get("depth"),
                "gold_label": gold,
                "question": str(rows[0].get("model_input", {}).get("question", "")),
                "predictions": predictions,
                "repair_methods": repair_methods,
                "errors": errors,
                "observation": observation,
            }
        )
    return analyses


def _render_case_analyses(analyses: list[dict[str, Any]]) -> str:
    lines = ["# Per-example smoke analysis", ""]
    for index, item in enumerate(analyses, start=1):
        lines.extend(
            [
                f"## {index}. `{item['example_id']}`",
                "",
                f"- Depth: {item['depth']}",
                f"- Gold label (evaluator-only): {item['gold_label']}",
                f"- Question: {item['question']}",
                f"- Predictions: {json.dumps(item['predictions'], ensure_ascii=False, sort_keys=True)}",
                f"- Repair methods: {item['repair_methods']}",
                f"- Explicit errors: {json.dumps(item['errors'], ensure_ascii=False, sort_keys=True)}",
                f"- Observation: {item['observation']}",
                "",
            ]
        )
    return "\n".join(lines)
