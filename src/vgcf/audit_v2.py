"""Offline, fail-loud auditor for VGCF-2.1 result artifacts."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from .data import render_model_input
from .experiment import load_config
from .protocol import (
    KNOWN_DATASET_SHA256,
    audit_sample_disjointness,
    load_quarantine,
    load_sample_manifest,
    load_theory_quarantine,
)

_SHARED_FORMALIZATION_FAMILY = {
    "constrained",
    "gate_cot",
    "repair_only",
    "vgcf2",
    "vgcf",
}
_SOLVER_ROUTES = {"solver_initial", "solver_repair"}
_FAILURE_ROUTES = {"method_error", "infrastructure_error"}
_HARD_GATED_METHODS = {"gate_cot", "vgcf2", "vgcf"}
_REPAIR_METHODS = {"repair_only", "vgcf2", "vgcf"}
_FORMALIZATION_METHODS = _SHARED_FORMALIZATION_FAMILY | {"plain"}


def audit_v2_run(
    result_path: str | Path, config_path: str | Path, project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    result = Path(result_path).resolve()
    manifest_path = result.with_suffix(".manifest.json")
    blocking: list[str] = []
    warnings: list[str] = []
    config, config_hash = load_config(config_path)
    rows = _read_jsonl(result)
    diagnostics = classify_run_failures(rows)
    if diagnostics["global_infrastructure_failure"]:
        blocking.append(
            "global infrastructure failure: every observed method result is an "
            "infrastructure_error and no successful model response exists"
        )
    elif diagnostics["partial_infrastructure_failure"]:
        blocking.append(
            "partial infrastructure failure: some method results are "
            "infrastructure_error"
        )
    manifest_exists = manifest_path.exists()
    if manifest_exists:
        manifest = _read_object(manifest_path)
    else:
        manifest = {}
        blocking.append(f"run manifest is missing: {manifest_path}")
    expected_count = config.sample_count * len(config.methods)
    if len(rows) != expected_count:
        blocking.append(f"expected {expected_count} rows, found {len(rows)}")
    if manifest_exists:
        if manifest.get("record_count") != len(rows):
            blocking.append("manifest record_count mismatch")
        if manifest.get("config_hash") != config_hash:
            blocking.append("manifest config hash mismatch")

        expected_dataset = _resolve(root, config.dataset_path)
        if Path(str(manifest.get("dataset_path", ""))).resolve() != expected_dataset:
            blocking.append("manifest dataset path mismatch")
        if manifest.get("dataset_sha256") != config.expected_dataset_sha256:
            blocking.append("manifest dataset hash mismatch")
        if manifest.get("phase") != config.phase:
            blocking.append("manifest phase mismatch")
        if manifest.get("dataset_split") == "meta-test" and config.phase != "final_test":
            blocking.append("non-final run used meta-test identity")

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("example_id"))].append(row)
    for example_id, group in grouped.items():
        methods = [str(row.get("method")) for row in group]
        if Counter(methods) != Counter(config.methods):
            blocking.append(f"method coverage mismatch for {example_id}")
    if len(grouped) != config.sample_count:
        blocking.append("unique example count mismatch")

    expected_split = {"dev": "meta-dev", "train": "meta-train", "final_test": "meta-test"}.get(
        config.phase
    )
    for row in rows:
        _audit_row(row, config, expected_split, blocking)

    if config.client.temperature != 0:
        blocking.append("temperature is not zero")
    if config.client.reasoning_effort != "none":
        blocking.append("reasoning_effort is not explicitly none")
    _audit_raw_files(rows, config, blocking)
    _audit_shared_responses(grouped, blocking)
    _audit_cot_fallbacks(grouped, blocking)

    if config.sample_manifest_path:
        sample_manifest = load_sample_manifest(_resolve(root, config.sample_manifest_path))
        if manifest_exists:
            selected = {
                str(item.get("example_id"))
                for item in manifest.get("selected_examples", [])
                if isinstance(item, Mapping)
            }
            expected = set(sample_manifest["sets"][config.sample_set]["example_ids"])
            if selected != expected:
                blocking.append("selected examples do not equal the frozen sample set")
            # The manifest embeds the complete audit performed before any request.
            sample_audit = manifest.get("sample_disjointness_audit")
            if not isinstance(sample_audit, Mapping) or sample_audit.get("status") != "pass":
                blocking.append("missing or failed sample-disjointness audit")
    if config.phase == "final_test":
        _audit_final_quarantine(rows, config, root, blocking)

    infrastructure_messages = {
        "global infrastructure failure: every observed method result is an "
        "infrastructure_error and no successful model response exists",
        "partial infrastructure failure: some method results are "
        "infrastructure_error",
    }
    method_or_protocol_failure = any(row.get("method_error") is True for row in rows) or any(
        message not in infrastructure_messages for message in blocking
    )
    diagnostics["method_or_protocol_failure"] = method_or_protocol_failure
    if diagnostics["global_infrastructure_failure"]:
        primary_diagnosis = "global_infrastructure_failure"
    elif diagnostics["partial_infrastructure_failure"]:
        primary_diagnosis = "partial_infrastructure_failure"
    elif method_or_protocol_failure:
        primary_diagnosis = "method_or_protocol_failure"
    else:
        primary_diagnosis = "none"
    diagnostics["primary_diagnosis"] = primary_diagnosis

    report = {
        "status": "pass" if not blocking else "fail",
        "result_path": str(result),
        "manifest_path": str(manifest_path),
        "expected_row_count": expected_count,
        "observed_row_count": len(rows),
        "diagnostics": diagnostics,
        "blocking": blocking,
        "warnings": warnings,
    }
    return report


def classify_run_failures(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Classify infrastructure failures without mistaking empty failures for responses."""

    infrastructure_rows = [row for row in rows if row.get("infrastructure_error") is True]
    successful_rows = [row for row in rows if _has_successful_model_response(row)]
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("example_id"))].append(row)
    all_infrastructure_examples = sorted(
        example_id
        for example_id, group in grouped.items()
        if group
        and all(row.get("infrastructure_error") is True for row in group)
        and not any(_has_successful_model_response(row) for row in group)
    )
    global_failure = bool(rows) and len(infrastructure_rows) == len(rows) and not successful_rows
    partial_failure = bool(infrastructure_rows) and not global_failure
    return {
        "primary_diagnosis": None,
        "global_infrastructure_failure": global_failure,
        "partial_infrastructure_failure": partial_failure,
        "method_or_protocol_failure": False,
        "observed_row_count": len(rows),
        "infrastructure_error_row_count": len(infrastructure_rows),
        "successful_model_response_row_count": len(successful_rows),
        "all_infrastructure_example_count": len(all_infrastructure_examples),
        "all_infrastructure_example_ids": all_infrastructure_examples,
    }


def _audit_row(
    row: Mapping[str, Any],
    config: Any,
    expected_split: str | None,
    blocking: list[str],
) -> None:
    tag = f"{row.get('example_id')}/{row.get('method')}"
    if row.get("phase") != config.phase:
        blocking.append(f"{tag}: row phase mismatch")
    if row.get("dataset_sha256") != config.expected_dataset_sha256:
        blocking.append(f"{tag}: row dataset hash mismatch")
    if expected_split and row.get("source_split") != expected_split:
        blocking.append(f"{tag}: example provenance mismatch")
    if config.phase != "final_test" and row.get("source_split") == "meta-test":
        blocking.append(f"{tag}: meta-test example in a non-final run")
    if row.get("model_id") != config.client.model_id:
        blocking.append(f"{tag}: model ID mismatch")
    model_input = row.get("model_input")
    if not isinstance(model_input, Mapping) or set(model_input) != {"theory", "question"}:
        blocking.append(f"{tag}: model input whitelist violation")
    route = row.get("route_source")
    method = row.get("method")
    answer_from_solver = row.get("answer_from_solver")
    if (route in _SOLVER_ROUTES) != (answer_from_solver is True):
        blocking.append(f"{tag}: answer_from_solver disagrees with route_source")
    if route == "solver_initial" and row.get("solver_executable") is not True:
        blocking.append(f"{tag}: solver_initial is not executable")
    if (
        route == "solver_initial"
        and method in _HARD_GATED_METHODS
        and row.get("hard_validator_pass") is not True
    ):
        blocking.append(f"{tag}: hard-gated solver_initial is not hard-valid")
    if route == "solver_repair":
        _audit_solver_repair(row, tag, blocking)
    if route == "cot_fallback" and row.get("fallback_used") is not True:
        blocking.append(f"{tag}: cot_fallback is not marked fallback_used")
    if route in _FAILURE_ROUTES and row.get("predicted_label") != "Error":
        blocking.append(f"{tag}: failure route was silently converted to a label")
    if (row.get("infrastructure_error") or row.get("method_error")) and row.get("predicted_label") == "Unknown":
        blocking.append(f"{tag}: failure was silently converted to Unknown")
    if row.get("infrastructure_error") and row.get("method_error"):
        blocking.append(f"{tag}: infrastructure_error and method_error both set")
    if getattr(config, "protocol_version", "VGCF-2.1") == "VGCF-2.2":
        _audit_v2_2_contract_fields(row, tag, blocking)
    tokens = row.get("tokens")
    if not isinstance(tokens, Mapping):
        blocking.append(f"{tag}: missing token accounting")
    else:
        prompt = tokens.get("prompt")
        completion = tokens.get("completion")
        total = tokens.get("total")
        if not all(isinstance(value, int) and value >= 0 for value in (prompt, completion, total)):
            blocking.append(f"{tag}: invalid token accounting")
        elif prompt + completion != total:
            blocking.append(f"{tag}: token total mismatch")
    paths = row.get("raw_output_paths")
    calls = row.get("call_count")
    if not isinstance(paths, list) or not isinstance(calls, int) or calls < len(paths) or calls < 1:
        blocking.append(f"{tag}: call/raw-path accounting mismatch")
    if not isinstance(row.get("latency_ms"), (int, float)) or row.get("latency_ms") < 0:
        blocking.append(f"{tag}: invalid latency")


def _audit_solver_repair(
    row: Mapping[str, Any], tag: str, blocking: list[str]
) -> None:
    method = row.get("method")
    if method not in _REPAIR_METHODS:
        blocking.append(f"{tag}: method cannot use solver_repair")
        return
    post = row.get("post_repair_result")
    if not isinstance(post, Mapping):
        blocking.append(f"{tag}: solver_repair has no post-repair result")
        return
    if row.get("repair_attempted") is not True:
        blocking.append(f"{tag}: solver_repair is not marked repair_attempted")
    if row.get("repair_improved") is not True:
        blocking.append(f"{tag}: solver_repair did not select an improved repair")
    if post.get("json_parse_valid") is not True or post.get("schema_valid") is not True:
        blocking.append(f"{tag}: solver_repair repair is not JSON/schema-valid")
    if post.get("solver_executable") is not True:
        blocking.append(f"{tag}: solver_repair repair is not executable")
    if method in {"vgcf2", "vgcf"} and post.get("hard_validator_pass") is not True:
        blocking.append(f"{tag}: Full VGCF-2 solver_repair is not hard-valid")

    selected_fields = (
        "json_parse_valid",
        "strict_schema_valid",
        "normalized_schema_valid",
        "schema_valid",
        "hard_validator_pass",
        "soft_issue_count",
        "solver_executable",
        "strict_solver_executable",
        "normalized_solver_executable",
        "parsed_structure",
    )
    for field in selected_fields:
        if row.get(field) != post.get(field):
            blocking.append(
                f"{tag}: selected {field} disagrees with post-repair result"
            )
    if row.get("shadow_solver_label") != post.get("solver_label"):
        blocking.append(f"{tag}: selected solver label disagrees with repair")
    if row.get("predicted_label") != post.get("solver_label"):
        blocking.append(f"{tag}: predicted label does not come from selected repair")


def _audit_raw_files(
    rows: list[dict[str, Any]], config: Any, blocking: list[str]
) -> None:
    seen: set[str] = set()
    forbidden_keys = ('"gold_label"', '"representation"', '"proofs"', '"answer"')
    for row in rows:
        model_input = row.get("model_input")
        expected_visible = (
            "NUMBERED_INPUT\n" + render_model_input(model_input)
            if isinstance(model_input, Mapping) and set(model_input) == {"theory", "question"}
            else None
        )
        for raw_name in row.get("raw_output_paths") or []:
            raw_path = str(raw_name)
            if raw_path in seen:
                continue
            seen.add(raw_path)
            path = Path(raw_path)
            if not path.exists():
                blocking.append(f"raw artifact is missing: {raw_path}")
                continue
            value = _read_object(path)
            content = value.get("content")
            if not isinstance(content, str) or not content.strip():
                blocking.append(f"raw artifact has empty visible content: {raw_path}")
            payload = value.get("request_payload")
            if not isinstance(payload, Mapping):
                blocking.append(f"raw artifact has no request payload: {raw_path}")
                continue
            if payload.get("model") != config.client.model_id or payload.get("temperature") != 0:
                blocking.append(f"raw artifact model/temperature mismatch: {raw_path}")
            if payload.get("reasoning_effort") != "none":
                blocking.append(f"raw artifact is not non-thinking: {raw_path}")
            messages = payload.get("messages")
            serialized = json.dumps(messages, ensure_ascii=False)
            if any(key in serialized for key in forbidden_keys):
                blocking.append(f"raw request contains an evaluator-only field key: {raw_path}")
            visible_messages = "\n".join(
                str(item.get("content", ""))
                for item in messages
                if isinstance(item, Mapping)
            ) if isinstance(messages, list) else ""
            if expected_visible and expected_visible not in visible_messages:
                blocking.append(f"raw request does not contain the exact whitelisted input: {raw_path}")
            if _contains_hidden_reasoning(value.get("provider_payload")):
                blocking.append(f"raw artifact contains hidden reasoning: {raw_path}")


def _audit_v2_2_contract_fields(
    row: Mapping[str, Any], tag: str, blocking: list[str]
) -> None:
    method = row.get("method")
    if row.get("latency_semantics") != "method_attributed_cache_sensitive_not_physical":
        blocking.append(f"{tag}: latency semantics are not explicit")
    if method == "cot_refine":
        valid = row.get("cot_refine_contract_valid")
        if row.get("infrastructure_error") is True:
            if valid is not None:
                blocking.append(f"{tag}: infrastructure failure claims a contract result")
        elif valid not in {True, False}:
            blocking.append(f"{tag}: missing CoT-Refine contract validity")
        if valid is False and row.get("method_error") is not True:
            blocking.append(f"{tag}: contract failure is not an explicit method_error")
        if valid is True and row.get("predicted_label") not in {
            "True",
            "False",
            "Unknown",
        }:
            blocking.append(f"{tag}: valid CoT-Refine contract has no valid label")
    elif row.get("cot_refine_contract_valid") is not None:
        blocking.append(f"{tag}: non-refine method has contract validity")

    if method in _FORMALIZATION_METHODS:
        required = (
            "strict_schema_valid",
            "normalized_schema_valid",
            "strict_solver_executable",
            "normalized_solver_executable",
        )
        if any(row.get(field) not in {True, False} for field in required):
            blocking.append(f"{tag}: missing strict/normalized structural metrics")
        if row.get("schema_valid") != row.get("normalized_schema_valid"):
            blocking.append(f"{tag}: schema_valid alias is not normalized_schema_valid")
        if row.get("solver_executable") != row.get("normalized_solver_executable"):
            blocking.append(
                f"{tag}: solver_executable alias is not normalized_solver_executable"
            )
        count = row.get("singleton_if_normalized_count")
        if not isinstance(count, int) or count < 0:
            blocking.append(f"{tag}: invalid singleton normalization count")

    expected_layers = _repair_regression_layers(row)
    actual_layers = tuple(row.get("repair_regression_layers") or [])
    if actual_layers != expected_layers:
        blocking.append(f"{tag}: repair regression layers are inconsistent")
    if row.get("repair_regression") is not bool(expected_layers):
        blocking.append(f"{tag}: repair_regression flag is inconsistent")


def _repair_regression_layers(row: Mapping[str, Any]) -> tuple[str, ...]:
    if row.get("repair_attempted") is not True:
        return ()
    pre = row.get("pre_repair_result")
    post = row.get("post_repair_result")
    if not isinstance(pre, Mapping) or not isinstance(post, Mapping):
        return ()
    fields = (
        ("json", "json_parse_valid"),
        ("strict_schema", "strict_schema_valid"),
        ("normalized_schema", "normalized_schema_valid"),
        ("strict_executable", "strict_solver_executable"),
        ("normalized_executable", "normalized_solver_executable"),
    )
    return tuple(
        layer
        for layer, field in fields
        if pre.get(field) is True and post.get(field) is False
    )


def _audit_shared_responses(
    grouped: Mapping[str, list[Mapping[str, Any]]], blocking: list[str]
) -> None:
    for example_id, rows in grouped.items():
        family = [row for row in rows if row.get("method") in _SHARED_FORMALIZATION_FAMILY]
        if len(family) <= 1:
            continue
        comparable: list[Mapping[str, Any]] = []
        for row in family:
            identity = (
                row.get("initial_response_hash"),
                row.get("initial_raw_output_path"),
                row.get("initial_cache_key"),
            )
            if all(isinstance(value, str) and value for value in identity):
                comparable.append(row)
                continue
            if row.get("infrastructure_error") is True and not any(identity):
                # No initial response exists to compare. A fresh run must retry it.
                continue
            method = row.get("method")
            if not isinstance(identity[0], str) or not identity[0]:
                blocking.append(f"{example_id}/{method}: missing initial response hash")
            if not isinstance(identity[1], str) or not identity[1]:
                blocking.append(f"{example_id}/{method}: missing initial raw path")
            if not isinstance(identity[2], str) or not identity[2]:
                blocking.append(f"{example_id}/{method}: missing initial cache identity")
        if len(comparable) <= 1:
            continue
        hashes = {row.get("initial_response_hash") for row in comparable}
        paths = {row.get("initial_raw_output_path") for row in comparable}
        keys = {row.get("initial_cache_key") for row in comparable}
        if len(hashes) != 1:
            blocking.append(f"{example_id}: formalization family did not share initial hash")
        if len(paths) != 1:
            blocking.append(f"{example_id}: formalization family did not share initial raw path")
        if len(keys) != 1:
            blocking.append(f"{example_id}: formalization family did not share cache identity")


def _audit_cot_fallbacks(
    grouped: Mapping[str, list[Mapping[str, Any]]], blocking: list[str]
) -> None:
    for example_id, rows in grouped.items():
        cot_rows = [row for row in rows if row.get("method") == "cot"]
        if not cot_rows:
            continue
        cot = cot_rows[0]
        expected = (
            cot.get("answer_response_hash"),
            cot.get("answer_raw_output_path"),
            cot.get("answer_cache_key"),
        )
        for row in rows:
            if row.get("route_source") != "cot_fallback":
                continue
            actual = (
                row.get("answer_response_hash"),
                row.get("answer_raw_output_path"),
                row.get("answer_cache_key"),
            )
            if actual != expected:
                blocking.append(f"{example_id}/{row.get('method')}: fallback did not reuse exact CoT response")


def _audit_final_quarantine(
    rows: list[dict[str, Any]], config: Any, root: Path, blocking: list[str]
) -> None:
    if not config.quarantine_path or not config.theory_quarantine_path:
        blocking.append("final run has no complete quarantine configuration")
        return
    questions = load_quarantine(_resolve(root, config.quarantine_path))
    theories = load_theory_quarantine(_resolve(root, config.theory_quarantine_path))
    for row in rows:
        if row.get("example_id") in questions or row.get("theory_id") in theories.theory_ids:
            blocking.append(f"quarantined final-test row: {row.get('example_id')}")


def _contains_hidden_reasoning(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in {"reasoning", "reasoning_content"} and isinstance(item, str) and item.strip():
                return True
            if _contains_hidden_reasoning(item):
                return True
    elif isinstance(value, list):
        return any(_contains_hidden_reasoning(item) for item in value)
    return False


def _has_successful_model_response(row: Mapping[str, Any]) -> bool:
    if isinstance(row.get("raw_model_response"), str) and row["raw_model_response"].strip():
        return True
    return any(
        isinstance(row.get(key), str) and bool(row[key])
        for key in (
            "initial_response_hash",
            "answer_response_hash",
            "initial_raw_output_path",
            "answer_raw_output_path",
        )
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"result line {line_number} is not an object")
            rows.append(value)
    return rows


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()
