"""Fail-closed VGCF-2.3.1 sensitivity and Final authorization rules.

This module is deliberately evaluator-side and offline.  It never constructs a
model client.  The same canonical functions are used by preflight, artifact
generation, authorization, and runtime so a hand-written PASS artifact cannot
replace recomputation from the frozen results and raw responses.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .data import file_sha256
from .errors import DatasetSchemaError
from .parsing import parse_cot_response

PROTOCOL = "VGCF-2.3.1"
REPAIR_BASE_COMMIT = "37b8ca5d5ac175aa6e0e4f1b9cc9c19472d181ce"
SENSITIVITY_ARM_ENV = "VGCF_COT4096_SENSITIVITY_ARM"
SENSITIVITY_ARM_VALUE = "VGCF2_3_1_COT4096_ONLY"
CONTRACT_ARM_ENV = "VGCF_CONTRACT12_ARM"
CONTRACT_ARM_VALUE = "VGCF2_3_1_SELECTED_BUDGET_ONLY"
FINAL_ARM_ENV = "VGCF_FINAL_TEST_ARM"
FINAL_ARM_VALUE = "VGCF2_3_1_FINAL_TEST"

AUTHORIZATION_POLICY_PATH = (
    "configs/final_authorization_policy_attempt2_v2_3_1.json"
)
CANDIDATE_CONTRACT_PATH = (
    "configs/final_candidate_contract_attempt2_v2_3_1_schema5.json"
)
LEGACY_CANDIDATE_CONTRACT_PATH = "configs/final_candidate_contract_v2_3_1.json"
ATTEMPT1_CANDIDATE_CONTRACT_PATH = (
    "configs/final_candidate_contract_v2_3_1_schema5.json"
)
SENSITIVITY_POLICY_PATH = (
    "configs/cot4096_sensitivity_attempt2_policy_v2_3_1.json"
)
SENSITIVITY_CONFIG_PATH = "configs/cot4096_sensitivity_attempt2_v2_3_1.json"
SENSITIVITY_MANIFEST_PATH = (
    "configs/cot4096_sensitivity_manifest_v2_3_1.json"
)
ATTEMPT1_REGISTRY_PATH = (
    "configs/cot4096_sensitivity_attempt1_registry_v2_3_1.json"
)
ATTEMPT1_REGISTRY_SHA256 = (
    "00457c45bf618a28b8dadc049fdffe391138d9b4eb7dce52000da4cf7ab693ec"
)
RUNTIME_PROFILE_PATH = "configs/lmstudio_runtime_profile_attempt2_v2_3_1.json"
BUDGET_DECISION_PATH = "configs/final_budget_decision_v2_3_1.json"
ACTIVE_FINAL_CONFIG_PATH = "configs/final_test_v2_3_1.json"
ACTIVE_FINAL_LOCK_PATH = "configs/final_test_lock_v2_3_1.json"
PRIMARY_HYPOTHESES_PATH = "configs/final_primary_hypotheses_v2_3_1.json"
FINAL_MANIFEST_PATH = "configs/final_300_manifest_v2_3_1.json"
PROTECTED_ARTIFACTS_PATH = "configs/protected_artifacts_v2_3_1.json"
CONTRACT_CANDIDATE_LOCK_PATH = (
    "configs/contract_candidate_lock_attempt2_v2_3_1.json"
)

FINAL_CANDIDATE_PATHS = {
    3000: "configs/final_test_3000_candidate_attempt2_v2_3_1.json",
    4096: "configs/final_test_4096_candidate_attempt2_v2_3_1.json",
}
CONTRACT_CANDIDATE_PATHS = {
    3000: "configs/contract_12_3000_candidate_attempt2_v2_3_1.json",
    4096: "configs/contract_12_4096_candidate_attempt2_v2_3_1.json",
}

DECISION_FUNCTION_NAME = "vgcf_2_3_1_operational_budget_decision"
DECISION_FUNCTION_VERSION = "3"
DECISION_INPUT_TOP_LEVEL_FIELDS = frozenset(
    {"case_ids", "case_reports", "criteria"}
)
DECISION_CASE_FIELDS = frozenset(
    {
        "example_id",
        "finish_reason",
        "finish_reason_present_and_not_length",
        "unique_legal_anchored_final_label",
        "parsed_label",
        "infrastructure_error",
        "hidden_reasoning",
        "contract_failure",
        "raw_path",
        "raw_sha256",
    }
)
DECISION_CRITERIA_FIELDS = frozenset(
    {
        "case_count",
        "finish_reason_present_and_not_length_count",
        "unique_legal_anchored_final_label_count",
        "infrastructure_error_count",
        "hidden_reasoning_count",
        "contract_failure_count",
    }
)
FORBIDDEN_DECISION_FIELDS = frozenset(
    {
        "gold_label",
        "accuracy",
        "final_label_correct",
        "answer",
        "proof",
        "formal_representation",
        "formal representation",
        "representation",
    }
)
BUDGET_DECISION_FIELDS = frozenset(
    {
        "schema_version",
        "protocol",
        "artifact_type",
        "status",
        "sensitivity_evidence",
        "operational_criteria",
        "decision_function",
        "decision_input_fields",
        "forbidden_fields_present",
        "decision_recomputed_from_raw_evidence",
        "gold_evaluation_disabled",
        "selected_max_tokens",
        "selected_final_candidate",
        "selected_contract_candidate",
        "model_calls_made",
        "network_calls_made",
        "decision_authorized",
    }
)


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Return the one canonical encoding used for generated authorization JSON."""

    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def tree_snapshot(path: Path) -> dict[str, Any]:
    """Hash every file by relative path and bytes, failing closed on absence."""

    if path.is_file():
        files = [path]
        base = path.parent
    elif path.is_dir():
        files = sorted(item for item in path.rglob("*") if item.is_file())
        base = path
    else:
        return {
            "exists": False,
            "file_count": 0,
            "total_bytes": 0,
            "tree_sha256": None,
        }
    rows: list[str] = []
    total_bytes = 0
    for item in files:
        rows.append(f"{item.relative_to(base).as_posix()}\t{file_sha256(item)}")
        total_bytes += item.stat().st_size
    return {
        "exists": True,
        "file_count": len(files),
        "total_bytes": total_bytes,
        "tree_sha256": hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest(),
    }


def audit_attempt1_preservation(root: Path) -> dict[str, Any]:
    """Recompute every registered Attempt-1 byte and tree identity."""

    root = root.resolve()
    registry_path = root / ATTEMPT1_REGISTRY_PATH
    if not registry_path.is_file():
        return {
            "status": "fail",
            "checks": {"registry_exists": False},
            "path": ATTEMPT1_REGISTRY_PATH,
        }
    try:
        registry = _read_object(registry_path)
    except Exception as exc:
        return {
            "status": "fail",
            "checks": {"registry_readable": False},
            "path": ATTEMPT1_REGISTRY_PATH,
            "error": str(exc),
        }
    artifact_checks: dict[str, bool] = {}
    for name, spec in registry.get("artifacts", {}).items():
        if not isinstance(spec, Mapping):
            artifact_checks[str(name)] = False
            continue
        relative = spec.get("path")
        path = root / str(relative)
        artifact_checks[str(name)] = (
            isinstance(relative, str)
            and path.is_file()
            and path.stat().st_size == spec.get("bytes")
            and file_sha256(path) == spec.get("sha256")
        )
    tree_paths = {
        "raw": "outputs/v2_3_1_cot4096_sensitivity/raw",
        "cache": "outputs/v2_3_1_cot4096_sensitivity/cache",
        "output_root": "outputs/v2_3_1_cot4096_sensitivity",
    }
    tree_checks: dict[str, bool] = {}
    tree_reports: dict[str, dict[str, Any]] = {}
    for name, relative in tree_paths.items():
        report = tree_snapshot(root / relative)
        tree_reports[name] = report
        expected = registry.get("trees", {}).get(name, {})
        tree_checks[name] = all(
            report.get(field) == expected.get(field)
            for field in ("file_count", "total_bytes", "tree_sha256")
        )
    frozen_input_checks = {
        str(name): isinstance(spec, Mapping)
        and isinstance(spec.get("path"), str)
        and (root / str(spec.get("path"))).is_file()
        and file_sha256(root / str(spec.get("path"))) == spec.get("sha256")
        for name, spec in registry.get("frozen_inputs", {}).items()
    }
    expected_reasons = [
        "four logical cases timed out",
        "eight 180-second transport attempts",
        "incomplete success raw/cache evidence",
        "sparse-vs-expanded config manifest mismatch",
        "context_length=4096 incompatible with the intended 4096-completion treatment",
    ]
    checks = {
        "registry_byte_identity": file_sha256(registry_path)
        == ATTEMPT1_REGISTRY_SHA256,
        "registry_invalid_forever": registry.get("attempt_id") == 1
        and registry.get("commit")
        == "53b5015a4f5026904e6a20a1888aece5fd2b0ebb"
        and registry.get("status") == "invalid_before_budget_decision"
        and registry.get("canonical_decision") is None
        and registry.get("decision_authorized") is False,
        "confirmed_reasons_exact": registry.get("confirmed_reasons")
        == expected_reasons,
        "bundle_hash_exact": registry.get("triage_bundle", {}).get("sha256")
        == "64003aba2f60245243c9e517b6ede06b0e6c1e37a665f8084995b5fc9246ee4f",
        "artifact_bytes": len(artifact_checks) == 5
        and all(artifact_checks.values()),
        "raw_cache_output_trees": len(tree_checks) == 3
        and all(tree_checks.values()),
        "frozen_inputs": len(frozen_input_checks) == 3
        and all(frozen_input_checks.values()),
        "no_downstream_authorization_registered": all(
            value is False
            for value in registry.get("downstream_authorization", {}).values()
        ),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "artifact_checks": artifact_checks,
        "tree_checks": tree_checks,
        "tree_reports": tree_reports,
        "frozen_input_checks": frozen_input_checks,
        "path": ATTEMPT1_REGISTRY_PATH,
        "sha256": file_sha256(registry_path),
        "model_calls_made": 0,
        "network_calls_made": 0,
    }


def audit_hash_contract(root: Path) -> dict[str, Any]:
    """Validate the post-fix candidate contract against current bytes."""

    path = root / CANDIDATE_CONTRACT_PATH
    if not path.is_file():
        return {
            "status": "fail",
            "checks": {"candidate_contract_exists": False},
            "path": CANDIDATE_CONTRACT_PATH,
        }
    contract = _read_object(path)
    authorization_policy_path = root / AUTHORIZATION_POLICY_PATH
    authorization_policy = (
        _read_object(authorization_policy_path)
        if authorization_policy_path.is_file()
        else {}
    )
    groups = (
        "artifact_sha256",
        "prompt_sha256",
        "inference_contract_sha256",
        "tooling_sha256",
    )
    reports: dict[str, dict[str, bool]] = {}
    for group_name in groups:
        expected = contract.get(group_name)
        if not isinstance(expected, Mapping) or not expected:
            reports[group_name] = {"mapping_present": False}
            continue
        reports[group_name] = {
            str(relative): _locked_file_matches(root, relative, expected_hash)
            for relative, expected_hash in expected.items()
        }
    checks = {
        "schema_protocol_status": contract.get("schema_version") == 3
        and contract.get("protocol") == PROTOCOL
        and contract.get("status")
        == "frozen_attempt2_infrastructure_before_sensitivity",
        "supersedes_attempt1_contract": contract.get("supersedes", {}).get(
            "path"
        )
        == ATTEMPT1_CANDIDATE_CONTRACT_PATH
        and contract.get("supersedes", {}).get("sha256")
        == file_sha256(root / ATTEMPT1_CANDIDATE_CONTRACT_PATH),
        "all_hashes_match": all(
            report and all(report.values()) for report in reports.values()
        ),
        "both_final_candidates_bound": set(FINAL_CANDIDATE_PATHS.values())
        <= set(contract.get("artifact_sha256", {})),
        "both_contract_candidates_bound": set(CONTRACT_CANDIDATE_PATHS.values())
        <= set(contract.get("artifact_sha256", {})),
        "contract_candidate_lock_bound": CONTRACT_CANDIDATE_LOCK_PATH
        in contract.get("artifact_sha256", {}),
        "authorization_policy_bound": AUTHORIZATION_POLICY_PATH
        in contract.get("artifact_sha256", {}),
        "runtime_profile_bound": RUNTIME_PROFILE_PATH
        in contract.get("artifact_sha256", {}),
        "attempt1_registry_bound": ATTEMPT1_REGISTRY_PATH
        in contract.get("artifact_sha256", {}),
        "schema4_activation_authority_retired": authorization_policy.get(
            "supersedes", {}
        ).get("schema_4_activation_authority")
        is False,
        "final_manifest_and_hypotheses_bound": {
            FINAL_MANIFEST_PATH,
            PRIMARY_HYPOTHESES_PATH,
        }
        <= set(contract.get("artifact_sha256", {})),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "groups": reports,
        "path": CANDIDATE_CONTRACT_PATH,
        "sha256": file_sha256(path),
    }


def audit_sensitivity(
    root: Path,
    *,
    preflight_only: bool = False,
    require_runtime_arm: bool = False,
    allow_downstream_artifacts: bool = False,
) -> dict[str, Any]:
    """Re-audit Attempt-2 without requiring five successful provider responses."""

    from .experiment import (
        canonical_normalized_config,
        config_semantic_sha256,
        normalize_config,
    )
    from .runtime_profile_v2_3_1 import verify_runtime_profile_binding

    root = root.resolve()
    required_paths = {
        "authorization_policy": root / AUTHORIZATION_POLICY_PATH,
        "sensitivity_policy": root / SENSITIVITY_POLICY_PATH,
        "config": root / SENSITIVITY_CONFIG_PATH,
        "sample_manifest": root / SENSITIVITY_MANIFEST_PATH,
        "attempt1_registry": root / ATTEMPT1_REGISTRY_PATH,
        "runtime_profile": root / RUNTIME_PROFILE_PATH,
    }
    missing_inputs = [
        name for name, path in required_paths.items() if not path.is_file()
    ]
    if missing_inputs:
        return {
            "schema_version": 3,
            "protocol": PROTOCOL,
            "attempt_id": 2,
            "execution_revision": "attempt2-infrastructure-v1",
            "mode": "preflight_only" if preflight_only else "post_run",
            "status": "fail" if preflight_only else "invalid",
            "run_status": "not_run",
            "blocking": [f"missing authorization inputs: {missing_inputs}"],
            "checks": {},
            "canonical_decision": None,
            "diagnostic_fallback_budget": None,
            "decision_authorized": False,
            "model_calls_made_by_auditor": 0,
            "network_calls_made_by_auditor": 0,
        }

    auth_policy = _read_object(required_paths["authorization_policy"])
    policy = _read_object(required_paths["sensitivity_policy"])
    config_path = required_paths["config"]
    config_value = _read_object(required_paths["config"])
    manifest = _read_object(required_paths["sample_manifest"])
    attempt1_registry = _read_object(required_paths["attempt1_registry"])
    attempt1_preservation = audit_attempt1_preservation(root)
    sensitivity_spec = policy.get("sensitivity", {})
    source_spec = policy.get("source_dev150", {})
    activation = auth_policy.get("activation", {})
    expected_ids = list(sensitivity_spec.get("case_ids", []))
    manifest_ids = list(
        manifest.get("sets", {})
        .get(str(sensitivity_spec.get("sample_set")), {})
        .get("example_ids", [])
    )
    result_path = root / str(sensitivity_spec.get("output_path", ""))
    run_manifest_path = root / str(sensitivity_spec.get("run_manifest_path", ""))
    raw_dir = root / str(sensitivity_spec.get("raw_dir", ""))
    cache_dir = root / str(sensitivity_spec.get("cache_dir", ""))
    attempts_dir = root / str(sensitivity_spec.get("attempts_dir", ""))
    terminal_dir = root / str(sensitivity_spec.get("terminal_dir", ""))
    audit_path = root / str(sensitivity_spec.get("audit_path", ""))
    source_results_path = root / str(source_spec.get("results_path", ""))
    source_manifest_path = root / str(source_spec.get("run_manifest_path", ""))
    active_config_path = root / str(
        activation.get("active_config_path", ACTIVE_FINAL_CONFIG_PATH)
    )
    active_lock_path = root / str(
        activation.get("active_lock_path", ACTIVE_FINAL_LOCK_PATH)
    )
    decision_path = root / str(
        activation.get("budget_decision_path", BUDGET_DECISION_PATH)
    )
    try:
        normalized_config = normalize_config(config_value)
        normalized_config_value = canonical_normalized_config(normalized_config)
        semantic_hash = config_semantic_sha256(normalized_config)
        config_normalization_error = None
    except (TypeError, ValueError) as exc:
        normalized_config = None
        normalized_config_value = None
        semantic_hash = None
        config_normalization_error = str(exc)
    try:
        if normalized_config is None:
            raise DatasetSchemaError("config normalization failed")
        verify_runtime_profile_binding(root, normalized_config)
        runtime_binding_error = None
    except DatasetSchemaError as exc:
        runtime_binding_error = str(exc)

    source_rows = (
        _read_jsonl(source_results_path) if source_results_path.is_file() else []
    )
    source_cot = {
        str(row.get("example_id")): row
        for row in source_rows
        if row.get("method") == "cot"
    }
    source_length_ids = sorted(
        example_id
        for example_id, row in source_cot.items()
        if _finish_reason_from_row(root, row) == "length"
    )
    hash_contract = audit_hash_contract(root)
    frozen_hashes = auth_policy.get("frozen_artifact_sha256", {})
    config_client = config_value.get("client", {})
    runtime_spec = policy.get("runtime_profile", {})
    checks = {
        "authorization_policy_schema": auth_policy.get("schema_version") == 3
        and auth_policy.get("protocol") == PROTOCOL
        and auth_policy.get("attempt_id") == 2
        and auth_policy.get("execution_revision")
        == "attempt2-infrastructure-v1"
        and auth_policy.get("status")
        == "frozen_attempt2_infrastructure_before_sensitivity",
        "candidate_contract_hash": hash_contract.get("status") == "pass"
        and auth_policy.get("candidate_contract", {}).get("path")
        == CANDIDATE_CONTRACT_PATH
        and auth_policy.get("candidate_contract", {}).get("bound_by")
        == f"{CANDIDATE_CONTRACT_PATH}:artifact_sha256",
        "frozen_artifact_hashes": isinstance(frozen_hashes, Mapping)
        and bool(frozen_hashes)
        and all(
            _locked_file_matches(root, relative, expected_hash)
            for relative, expected_hash in frozen_hashes.items()
        ),
        "attempt2_policy_schema": policy.get("schema_version") == 2
        and policy.get("protocol") == PROTOCOL
        and policy.get("attempt_id") == 2
        and policy.get("execution_revision") == "attempt2-infrastructure-v1",
        "attempt1_registry_frozen_invalid": attempt1_registry.get("attempt_id") == 1
        and attempt1_registry.get("commit")
        == "53b5015a4f5026904e6a20a1888aece5fd2b0ebb"
        and attempt1_registry.get("status") == "invalid_before_budget_decision"
        and attempt1_registry.get("canonical_decision") is None
        and attempt1_registry.get("decision_authorized") is False
        and file_sha256(required_paths["attempt1_registry"])
        == policy.get("attempt1_registry", {}).get("sha256"),
        "attempt1_artifacts_byte_identical": attempt1_preservation.get("status")
        == "pass",
        "source_results_hash": source_results_path.is_file()
        and file_sha256(source_results_path) == source_spec.get("results_sha256"),
        "source_run_manifest_hash": source_manifest_path.is_file()
        and file_sha256(source_manifest_path)
        == source_spec.get("run_manifest_sha256"),
        "source_finish_reason_length_exact_set": source_length_ids
        == sorted(expected_ids),
        "config_file_sha256": file_sha256(required_paths["config"])
        == sensitivity_spec.get("config_file_sha256"),
        "config_semantic_sha256": config_normalization_error is None
        and semantic_hash == sensitivity_spec.get("config_semantic_sha256"),
        "config_strict_normalization": config_normalization_error is None,
        "sample_manifest_hash": file_sha256(required_paths["sample_manifest"])
        == sensitivity_spec.get("sample_manifest_sha256"),
        "exact_five_ids": expected_ids == manifest_ids
        and len(expected_ids) == len(set(expected_ids)) == 5,
        "cot_prompt_and_method_frozen": config_value.get("methods") == ["cot"]
        and config_value.get("sample_count") == 5
        and config_value.get("prompt_directory") == "configs/prompts",
        "attempt2_identity": config_value.get("attempt_id") == 2
        and config_value.get("execution_revision")
        == "attempt2-infrastructure-v1",
        "runtime_profile_binding": runtime_binding_error is None
        and runtime_spec.get("path") == RUNTIME_PROFILE_PATH
        and runtime_spec.get("sha256")
        == file_sha256(required_paths["runtime_profile"]),
        "transport_policy": config_client.get("timeout_seconds") == 360
        and config_client.get("max_retries") == 1
        and config_client.get("retry_backoff_seconds") == 1.0
        and config_client.get("fresh_cache_required") is True
        and config_client.get("use_cache") is True,
        "model_generation_settings": config_client.get("model_id")
        == "qwen3.5-9b"
        and config_client.get("temperature") == 0
        and config_client.get("reasoning_effort") == "none"
        and config_client.get("max_tokens") == 4096,
        "gold_evaluation_disabled": config_value.get("gold_evaluation") is False,
        "paths_match_policy": config_value.get("output_path")
        == sensitivity_spec.get("output_path")
        and all(
            config_client.get(field) == sensitivity_spec.get(field)
            for field in ("raw_dir", "cache_dir", "attempts_dir", "terminal_dir")
        ),
        "attempt1_cache_not_reused": config_client.get("cache_dir")
        != "outputs/v2_3_1_cot4096_sensitivity/cache",
        "active_final_absent": not active_config_path.exists()
        and not active_lock_path.exists(),
        "budget_decision_absent_before_selection": not decision_path.exists(),
        "strict_descendant_of_attempt1_commit": _is_strict_descendant(
            root, "53b5015a4f5026904e6a20a1888aece5fd2b0ebb"
        ),
        "tracked_worktree_clean": _tracked_worktree_clean(root),
    }
    if require_runtime_arm:
        checks["python_runtime_arm"] = (
            os.environ.get(SENSITIVITY_ARM_ENV) == SENSITIVITY_ARM_VALUE
            and auth_policy.get("sensitivity_arm", {}).get("environment_variable")
            == SENSITIVITY_ARM_ENV
            and auth_policy.get("sensitivity_arm", {}).get("value")
            == SENSITIVITY_ARM_VALUE
        )
    freshness_paths = (
        result_path,
        run_manifest_path,
        raw_dir,
        cache_dir,
        attempts_dir,
        terminal_dir,
        audit_path,
    )
    if preflight_only:
        checks["fresh_attempt2_output_evidence_paths"] = not any(
            path.exists() for path in freshness_paths
        )
    if allow_downstream_artifacts:
        checks["active_final_absent"] = True
        checks["budget_decision_absent_before_selection"] = True
    effective_checks = dict(checks)
    if preflight_only and not require_runtime_arm:
        for name in (
            "strict_descendant_of_attempt1_commit",
            "tracked_worktree_clean",
        ):
            effective_checks.pop(name, None)
    if allow_downstream_artifacts:
        effective_checks.pop("active_final_absent", None)
        effective_checks.pop("budget_decision_absent_before_selection", None)
    blocking = [name for name, passed in effective_checks.items() if not passed]
    base = {
        "schema_version": 3,
        "protocol": PROTOCOL,
        "attempt_id": 2,
        "execution_revision": "attempt2-infrastructure-v1",
        "mode": "preflight_only" if preflight_only else "post_run",
        "checks": checks,
        "blocking": blocking,
        "source_length_case_ids": source_length_ids,
        "frozen_case_ids": expected_ids,
        "run_status": "not_run" if not result_path.exists() else "present",
        "config_normalization_error": config_normalization_error,
        "runtime_profile_binding_error": runtime_binding_error,
        "attempt1_preservation": attempt1_preservation,
        "canonical_decision": None,
        "diagnostic_fallback_budget": None,
        "decision_authorized": False,
        "model_calls_made_by_auditor": 0,
        "network_calls_made_by_auditor": 0,
    }
    if preflight_only:
        return {
            **base,
            "status": "pass" if not blocking else "fail",
            "decision": "not_run",
            "execution_ready_after_attempt2_commit": all(checks.values()),
        }
    if not result_path.is_file():
        return {
            **base,
            "status": "pending" if not blocking else "invalid",
            "run_status": "not_run",
            "blocking": [*blocking, "sensitivity result is missing"],
            "decision": "no_budget_decision",
        }

    try:
        rows = _read_jsonl(result_path)
    except Exception as exc:
        rows = []
        blocking.append(f"sensitivity results are unreadable: {exc}")
    expected_pairs = [(example_id, "cot") for example_id in expected_ids]
    actual_pairs = [(row.get("example_id"), row.get("method")) for row in rows]
    if len(rows) != 5 or sorted(actual_pairs) != sorted(expected_pairs):
        blocking.append("sensitivity rows are not exactly one CoT row per frozen ID")
    if len(set(actual_pairs)) != len(actual_pairs):
        blocking.append("sensitivity rows contain duplicate method/example pairs")
    by_id = {
        str(row.get("example_id")): row
        for row in rows
        if row.get("method") == "cot"
    }
    audited_cases = [
        _audit_attempt2_case_evidence(
            root,
            example_id,
            by_id.get(example_id),
            raw_dir=raw_dir,
            cache_dir=cache_dir,
            attempts_dir=attempts_dir,
            terminal_dir=terminal_dir,
        )
        for example_id in expected_ids
    ]
    case_reports = [item["case_report"] for item in audited_cases]
    evidence_reports = [item["evidence"] for item in audited_cases]
    for evidence in evidence_reports:
        blocking.extend(str(item) for item in evidence.get("blocking", []))
    criteria = criteria_from_case_reports(case_reports)
    decision_input = {
        "case_ids": expected_ids,
        "case_reports": case_reports,
        "criteria": criteria,
    }
    forbidden = sorted(_forbidden_fields_present(decision_input))
    try:
        diagnostic_budget = canonical_selected_budget(auth_policy, decision_input)
    except (TypeError, ValueError) as exc:
        diagnostic_budget = None
        blocking.append(f"canonical decision failed: {exc}")

    raw_tree = tree_snapshot(raw_dir)
    cache_tree = tree_snapshot(cache_dir)
    attempt_tree = tree_snapshot(attempts_dir)
    terminal_tree = tree_snapshot(terminal_dir)
    actual_raw = _relative_file_set(root, raw_dir)
    actual_cache = _relative_file_set(root, cache_dir)
    actual_attempts = _relative_file_set(root, attempts_dir)
    actual_terminals = _relative_file_set(root, terminal_dir)
    referenced_raw = {
        str(item["success_raw_path"])
        for item in evidence_reports
        if item.get("success_raw_path")
    }
    referenced_cache = {
        str(item["success_cache_path"])
        for item in evidence_reports
        if item.get("success_cache_path")
    }
    referenced_attempts = {
        str(path)
        for item in evidence_reports
        for path in item.get("attempt_paths", [])
    }
    referenced_terminals = {
        str(item["terminal_path"])
        for item in evidence_reports
        if item.get("terminal_path")
    }
    success_count = sum(item.get("terminal_outcome") == "success" for item in evidence_reports)
    terminal_attempt_sum = sum(
        int(item.get("transport_attempt_count", 0)) for item in evidence_reports
    )
    evidence_checks = {
        "exactly_five_terminal_artifacts": len(actual_terminals) == 5
        and referenced_terminals == actual_terminals,
        "terminal_attempt_hash_chains_complete": bool(evidence_reports)
        and all(not item.get("blocking") for item in evidence_reports),
        "success_raw_count": len(actual_raw) == success_count
        and referenced_raw == actual_raw,
        "success_cache_count": len(actual_cache) == success_count
        and referenced_cache == actual_cache,
        "attempt_count_matches_terminals": len(actual_attempts)
        == terminal_attempt_sum
        and referenced_attempts == actual_attempts,
        "logical_request_ids_unique": len(
            {
                item.get("logical_request_id")
                for item in evidence_reports
                if item.get("logical_request_id")
            }
        )
        == 5,
    }
    blocking.extend(
        name for name, passed in evidence_checks.items() if not passed
    )

    try:
        run_manifest = (
            _read_object(run_manifest_path) if run_manifest_path.is_file() else {}
        )
        run_manifest_read_error = None
    except Exception as exc:
        run_manifest = {}
        run_manifest_read_error = str(exc)
    manifest_ids_from_run = [
        item.get("example_id")
        for item in run_manifest.get("selected_examples", [])
        if isinstance(item, Mapping)
    ]
    manifest_config_error: str | None = None
    manifest_normalized: dict[str, Any] | None = None
    try:
        manifest_normalized = canonical_normalized_config(
            normalize_config(run_manifest.get("config", {}))
        )
    except (TypeError, ValueError) as exc:
        manifest_config_error = str(exc)
    current_head = _git_output(root, ["rev-parse", "HEAD"])
    config_file_hash = file_sha256(config_path)
    row_file_hashes = {
        str(row.get("config_file_sha256", row.get("config_hash"))) for row in rows
    }
    row_semantic_hashes = {
        str(row.get("config_semantic_sha256")) for row in rows
    }
    row_git_commits = {str(row.get("git_commit")) for row in rows}
    manifest_checks = {
        "run_manifest_exists": run_manifest_path.is_file()
        and run_manifest_read_error is None,
        "config_file_sha256_independent": run_manifest.get("config_file_sha256")
        == config_file_hash
        and run_manifest.get("config_hash") == config_file_hash
        and row_file_hashes == {config_file_hash},
        "config_semantic_sha256_independent": semantic_hash is not None
        and run_manifest.get("config_semantic_sha256") == semantic_hash
        and row_semantic_hashes == {semantic_hash},
        "manifest_config_shared_normalizer": manifest_config_error is None
        and manifest_normalized == normalized_config_value,
        "canonical_normalized_config_stored": run_manifest.get("normalized_config")
        == normalized_config_value,
        "run_identity": run_manifest.get("record_count") == 5
        and run_manifest.get("selected_example_count") == 5
        and run_manifest.get("sample_set") == "cot4096_length_5"
        and manifest_ids_from_run == expected_ids
        and run_manifest.get("gold_evaluation") is False
        and run_manifest.get("git_commit") == current_head
        and row_git_commits == {current_head},
    }
    blocking.extend(name for name, passed in manifest_checks.items() if not passed)
    if forbidden:
        blocking.append("forbidden fields entered canonical decision input")
    blocking = list(dict.fromkeys(blocking))
    complete = not blocking
    canonical_decision = (
        {"selected_max_tokens": diagnostic_budget}
        if complete and diagnostic_budget in (3000, 4096)
        else None
    )
    return {
        **base,
        "status": "complete" if complete else "invalid",
        "run_status": "audited",
        "blocking": blocking,
        "result_path": result_path.relative_to(root).as_posix(),
        "result_sha256": file_sha256(result_path),
        "run_manifest_path": run_manifest_path.relative_to(root).as_posix(),
        "run_manifest_sha256": (
            file_sha256(run_manifest_path) if run_manifest_path.is_file() else None
        ),
        "raw_tree": raw_tree,
        "cache_tree": cache_tree,
        "attempt_tree": attempt_tree,
        "terminal_tree": terminal_tree,
        "case_reports": case_reports,
        "evidence_reports": evidence_reports,
        "evidence_checks": evidence_checks,
        "manifest_checks": manifest_checks,
        "manifest_config_error": manifest_config_error,
        "run_manifest_read_error": run_manifest_read_error,
        "criteria": criteria,
        "decision_input_fields": {
            "top_level": sorted(DECISION_INPUT_TOP_LEVEL_FIELDS),
            "case_report": sorted(DECISION_CASE_FIELDS),
            "criteria": sorted(DECISION_CRITERIA_FIELDS),
        },
        "forbidden_fields_present": forbidden,
        "decision_recomputed_from_raw_evidence": complete,
        "gold_evaluation_disabled": config_value.get("gold_evaluation") is False,
        "canonical_decision": canonical_decision,
        "diagnostic_fallback_budget": diagnostic_budget if not complete else None,
        "decision_authorized": complete and canonical_decision is not None,
        "selected_max_tokens_recomputed": (
            diagnostic_budget if complete else None
        ),
        "decision": (
            f"select_final_{diagnostic_budget}"
            if complete and diagnostic_budget is not None
            else "no_budget_decision"
        ),
    }


def audit_sensitivity_case(
    root: Path,
    example_id: str,
    row: Mapping[str, Any] | None,
    expected_raw_dir: Path,
) -> dict[str, Any]:
    """Derive one operational case exclusively from whitelisted row/raw fields."""

    default = {
        "example_id": example_id,
        "finish_reason": None,
        "finish_reason_present_and_not_length": False,
        "unique_legal_anchored_final_label": False,
        "parsed_label": None,
        "infrastructure_error": True,
        "hidden_reasoning": False,
        "contract_failure": True,
        "raw_path": None,
        "raw_sha256": None,
    }
    if row is None or set(row).isdisjoint({"example_id", "method"}):
        return default
    infrastructure_error = row.get("infrastructure_error") is not False
    raw_paths = row.get("raw_output_paths")
    raw_path: Path | None = None
    if isinstance(raw_paths, list) and len(raw_paths) == 1:
        candidate = Path(str(raw_paths[0]))
        candidate = (
            candidate.resolve()
            if candidate.is_absolute()
            else (root / candidate).resolve()
        )
        try:
            candidate.relative_to(expected_raw_dir.resolve())
        except ValueError:
            candidate = Path()
        if candidate.is_file():
            raw_path = candidate
    if raw_path is None:
        return {**default, "infrastructure_error": True}
    raw = _read_object(raw_path)
    finish_reason = _provider_finish_reason(raw)
    hidden_reasoning = _contains_hidden_reasoning(raw.get("provider_payload"))
    content = raw.get("content")
    parsed_label: str | None = None
    if isinstance(content, str):
        try:
            parsed_label = parse_cot_response(content).label
        except Exception:
            parsed_label = None
    anchored_valid = parsed_label in {"True", "False", "Unknown"}
    contract_failure = (
        not anchored_valid
        or row.get("method_error") is not False
        or row.get("predicted_label") != parsed_label
        or row.get("example_id") != example_id
        or row.get("method") != "cot"
    )
    return {
        "example_id": example_id,
        "finish_reason": finish_reason,
        "finish_reason_present_and_not_length": isinstance(finish_reason, str)
        and bool(finish_reason)
        and finish_reason != "length",
        "unique_legal_anchored_final_label": anchored_valid,
        "parsed_label": parsed_label,
        "infrastructure_error": infrastructure_error,
        "hidden_reasoning": hidden_reasoning,
        "contract_failure": contract_failure,
        "raw_path": raw_path.relative_to(root).as_posix(),
        "raw_sha256": file_sha256(raw_path),
    }


def _audit_attempt2_case_evidence(
    root: Path,
    example_id: str,
    row: Mapping[str, Any] | None,
    *,
    raw_dir: Path,
    cache_dir: Path,
    attempts_dir: Path,
    terminal_dir: Path,
) -> dict[str, Any]:
    failed_case = {
        "example_id": example_id,
        "finish_reason": None,
        "finish_reason_present_and_not_length": False,
        "unique_legal_anchored_final_label": False,
        "parsed_label": None,
        "infrastructure_error": True,
        "hidden_reasoning": False,
        "contract_failure": True,
        "raw_path": None,
        "raw_sha256": None,
    }
    evidence: dict[str, Any] = {
        "example_id": example_id,
        "logical_request_id": None,
        "terminal_path": None,
        "terminal_sha256": None,
        "terminal_outcome": None,
        "transport_attempt_count": 0,
        "attempt_paths": [],
        "success_raw_path": None,
        "success_cache_path": None,
        "physical_elapsed_ms": 0.0,
        "error_category": None,
        "provider_usage_available": False,
        "blocking": [],
    }
    blocking: list[str] = evidence["blocking"]

    def fail(message: str) -> None:
        blocking.append(f"{example_id}: {message}")

    if not isinstance(row, Mapping):
        fail("result row is missing")
        return {"case_report": failed_case, "evidence": evidence}
    terminal_path = _resolved_evidence_path(
        root, row.get("terminal_evidence_path"), terminal_dir
    )
    if terminal_path is None or not terminal_path.is_file():
        fail("terminal evidence is missing or outside the frozen terminal directory")
        return {"case_report": failed_case, "evidence": evidence}
    evidence["terminal_path"] = terminal_path.relative_to(root).as_posix()
    evidence["terminal_sha256"] = file_sha256(terminal_path)
    try:
        terminal = _read_object(terminal_path)
    except Exception as exc:
        fail(f"terminal evidence is unreadable: {exc}")
        return {"case_report": failed_case, "evidence": evidence}
    required_terminal = {
        "schema_version",
        "logical_request_id",
        "request_tag",
        "cache_key",
        "attempt_artifacts",
        "transport_attempt_count",
        "total_physical_elapsed_ms",
        "terminal_outcome",
        "success_raw_path",
        "success_raw_sha256",
        "success_cache_path",
        "success_cache_sha256",
        "terminal_error_category",
        "provider_usage_available",
    }
    if set(terminal) != required_terminal:
        fail("terminal fields do not match the append-only schema")
    logical_request_id = terminal.get("logical_request_id")
    if not isinstance(logical_request_id, str) or not logical_request_id:
        fail("terminal logical_request_id is invalid")
        logical_request_id = "invalid"
    evidence["logical_request_id"] = logical_request_id
    expected_terminal = (terminal_dir / f"{logical_request_id}.json").resolve()
    if terminal_path != expected_terminal:
        fail("terminal path does not match logical_request_id")
    if row.get("logical_request_ids") != [logical_request_id]:
        fail("result logical_request_ids do not bind the terminal")
    if row.get("terminal_evidence_paths") not in (
        [str(terminal_path)],
        [terminal_path.as_posix()],
        [terminal_path.relative_to(root).as_posix()],
    ):
        fail("result terminal_evidence_paths do not bind exactly one terminal")
    if terminal.get("schema_version") != 1:
        fail("terminal schema_version is not 1")
    if not isinstance(terminal.get("request_tag"), str):
        fail("terminal request_tag is invalid")
    if terminal.get("request_tag") != f"cot-{example_id}":
        fail("terminal request_tag does not bind the frozen case")
    cache_key = terminal.get("cache_key")
    if not isinstance(cache_key, str) or re.fullmatch(r"[0-9a-f]{64}", cache_key) is None:
        fail("terminal cache_key is invalid")
    elif logical_request_id != _expected_logical_request_id(
        str(terminal.get("request_tag")), cache_key
    ):
        fail("terminal logical_request_id does not reproduce request_tag/cache_key")
    attempt_specs = terminal.get("attempt_artifacts")
    if not isinstance(attempt_specs, list) or not 1 <= len(attempt_specs) <= 2:
        fail("terminal must bind one or two physical attempts")
        attempt_specs = []
    terminal_count = terminal.get("transport_attempt_count")
    if terminal_count != len(attempt_specs):
        fail("terminal transport_attempt_count does not match attempt artifacts")
    evidence["transport_attempt_count"] = len(attempt_specs)
    attempts: list[dict[str, Any]] = []
    attempt_paths: list[str] = []
    for index, spec in enumerate(attempt_specs, start=1):
        if not isinstance(spec, Mapping):
            fail(f"attempt-{index:02d} binding is not an object")
            continue
        attempt_path = _resolved_evidence_path(
            root, spec.get("path"), attempts_dir / logical_request_id
        )
        if attempt_path is None or not attempt_path.is_file():
            fail(f"attempt-{index:02d} is missing or cross-bound")
            continue
        relative_attempt = attempt_path.relative_to(root).as_posix()
        attempt_paths.append(relative_attempt)
        expected_attempt_path = (
            attempts_dir / logical_request_id / f"attempt-{index:02d}.json"
        ).resolve()
        if attempt_path != expected_attempt_path:
            fail(f"attempt-{index:02d} path does not match its index")
        if spec.get("sha256") != file_sha256(attempt_path):
            fail(f"attempt-{index:02d} hash does not match terminal binding")
        try:
            attempt = _read_object(attempt_path)
        except Exception as exc:
            fail(f"attempt-{index:02d} is unreadable: {exc}")
            continue
        attempts.append(attempt)
        required_attempt = {
            "schema_version",
            "logical_request_id",
            "request_tag",
            "cache_key",
            "attempt_index",
            "max_attempts",
            "started_utc",
            "finished_utc",
            "elapsed_ms",
            "provider",
            "model",
            "endpoint",
            "timeout_seconds",
            "max_tokens",
            "outcome_category",
            "exception_class",
            "exception_message",
            "response_received",
            "valid_response_body",
            "retry_eligible",
            "request_payload_sha256",
        }
        if set(attempt) != required_attempt:
            fail(f"attempt-{index:02d} fields do not match the evidence schema")
        identity_checks = (
            attempt.get("schema_version") == 1,
            attempt.get("logical_request_id") == logical_request_id,
            attempt.get("request_tag") == terminal.get("request_tag"),
            attempt.get("cache_key") == cache_key,
            attempt.get("attempt_index") == index,
            attempt.get("max_attempts") == 2,
            attempt.get("provider") == "lmstudio",
            attempt.get("model") == "qwen3.5-9b",
            attempt.get("endpoint")
            == "http://127.0.0.1:1234/v1/chat/completions",
            attempt.get("timeout_seconds") == 360,
            attempt.get("max_tokens") == 4096,
        )
        if not all(identity_checks):
            fail(f"attempt-{index:02d} identity/runtime fields changed")
        elapsed = attempt.get("elapsed_ms")
        if (
            not isinstance(elapsed, (int, float))
            or isinstance(elapsed, bool)
            or elapsed < 0
        ):
            fail(f"attempt-{index:02d} elapsed_ms is invalid")
        try:
            started = datetime.fromisoformat(str(attempt.get("started_utc")))
            finished = datetime.fromisoformat(str(attempt.get("finished_utc")))
            if (
                started.utcoffset() is None
                or finished.utcoffset() is None
                or started.utcoffset().total_seconds() != 0
                or finished.utcoffset().total_seconds() != 0
                or finished < started
            ):
                raise ValueError("finished before started")
        except ValueError:
            fail(f"attempt-{index:02d} timestamps are invalid")
        payload_sha = attempt.get("request_payload_sha256")
        if not isinstance(payload_sha, str) or re.fullmatch(
            r"[0-9a-f]{64}", payload_sha
        ) is None:
            fail(f"attempt-{index:02d} request payload hash is invalid")
        outcome = attempt.get("outcome_category")
        retryable_categories = {
            "connection_refused",
            "connection_reset",
            "http_429",
            "http_502",
            "http_503",
            "http_504",
        }
        allowed_categories = retryable_categories | {
            "success",
            "timeout",
            "json_decode_error",
            "provider_schema_error",
            "empty_visible_content",
            "hidden_reasoning_contamination",
            "model_error",
            "transport_error",
            "unexpected_error",
        }
        if outcome not in allowed_categories:
            fail(f"attempt-{index:02d} outcome category is unknown")
        if outcome == "success":
            if (
                attempt.get("exception_class") is not None
                or attempt.get("exception_message") is not None
                or attempt.get("response_received") is not True
                or attempt.get("valid_response_body") is not True
                or attempt.get("retry_eligible") is not False
            ):
                fail(f"attempt-{index:02d} success metadata is inconsistent")
        elif not (
            isinstance(attempt.get("exception_class"), str)
            and isinstance(attempt.get("exception_message"), str)
            and isinstance(attempt.get("response_received"), bool)
            and isinstance(attempt.get("valid_response_body"), bool)
            and isinstance(attempt.get("retry_eligible"), bool)
        ):
            fail(f"attempt-{index:02d} failure metadata is incomplete")
        expected_retry_eligible = (
            outcome in retryable_categories
            and attempt.get("valid_response_body") is False
        )
        if attempt.get("retry_eligible") is not expected_retry_eligible:
            fail(f"attempt-{index:02d} retry classification is inconsistent")
    evidence["attempt_paths"] = attempt_paths
    row_attempt_paths = row.get("transport_attempt_paths")
    expected_attempt_paths_absolute = [
        str((root / path).resolve()) for path in attempt_paths
    ]
    if row_attempt_paths not in (attempt_paths, expected_attempt_paths_absolute):
        fail("result transport_attempt_paths do not match terminal order")
    if row.get("transport_attempt_count") != len(attempt_paths):
        fail("result transport_attempt_count does not match terminal")
    payload_hashes = {
        attempt.get("request_payload_sha256") for attempt in attempts
    }
    if len(payload_hashes) != 1:
        fail("physical attempts do not bind one request payload")
    if payload_hashes != {cache_key}:
        fail("request payload hash does not reproduce cache_key")
    for index, attempt in enumerate(attempts[:-1], start=1):
        if (
            attempt.get("retry_eligible") is not True
            or attempt.get("valid_response_body") is not False
            or attempt.get("outcome_category")
            not in {
                "connection_refused",
                "connection_reset",
                "http_429",
                "http_502",
                "http_503",
                "http_504",
            }
        ):
            fail(f"attempt-{index:02d} was not eligible for the sole retry")
    if any(
        attempt.get("outcome_category") == "success" for attempt in attempts[:-1]
    ):
        fail("a successful attempt was followed by another physical attempt")
    if (
        len(attempts) == 1
        and attempts[0].get("retry_eligible") is True
    ):
        fail("a retry-eligible first attempt is missing its sole retry")
    summed_elapsed = sum(
        float(attempt.get("elapsed_ms", 0.0))
        for attempt in attempts
        if isinstance(attempt.get("elapsed_ms"), (int, float))
        and not isinstance(attempt.get("elapsed_ms"), bool)
    )
    terminal_elapsed = terminal.get("total_physical_elapsed_ms")
    if not _same_number(terminal_elapsed, summed_elapsed):
        fail("terminal physical elapsed does not equal attempt sum")
    if not _same_number(row.get("physical_elapsed_ms"), summed_elapsed):
        fail("result physical_elapsed_ms does not equal attempt sum")
    evidence["physical_elapsed_ms"] = summed_elapsed
    terminal_outcome = terminal.get("terminal_outcome")
    evidence["terminal_outcome"] = terminal_outcome
    last_outcome = attempts[-1].get("outcome_category") if attempts else None
    if terminal_outcome == "success":
        if last_outcome != "success" or terminal.get("terminal_error_category") is not None:
            fail("success terminal does not end in a successful attempt")
        raw_path = _resolved_evidence_path(
            root, terminal.get("success_raw_path"), raw_dir
        )
        cache_path = _resolved_evidence_path(
            root, terminal.get("success_cache_path"), cache_dir
        )
        if raw_path is None or not raw_path.is_file():
            fail("success terminal raw artifact is missing or cross-bound")
        if cache_path is None or not cache_path.is_file():
            fail("success terminal cache artifact is missing or cross-bound")
        if raw_path is not None and raw_path.is_file():
            evidence["success_raw_path"] = raw_path.relative_to(root).as_posix()
            if terminal.get("success_raw_sha256") != file_sha256(raw_path):
                fail("success raw hash does not match terminal")
        if cache_path is not None and cache_path.is_file():
            evidence["success_cache_path"] = cache_path.relative_to(root).as_posix()
            if terminal.get("success_cache_sha256") != file_sha256(cache_path):
                fail("success cache hash does not match terminal")
        row_raw_paths = row.get("raw_output_paths")
        valid_row_raw = (
            raw_path is not None
            and row_raw_paths
            in ([str(raw_path)], [raw_path.relative_to(root).as_posix()])
        )
        if not valid_row_raw or row.get("answer_cache_key") != cache_key:
            fail("success result does not bind exactly one raw and cache key")
        if row.get("infrastructure_error") is not False:
            fail("success terminal is marked as an infrastructure error")
        expected_result_error = (
            "output_contract_failure"
            if row.get("method_error") is True
            else None
        )
        if row.get("error_category") != expected_result_error:
            fail("success terminal result error_category is inconsistent")
        if terminal.get("provider_usage_available") is not row.get(
            "provider_usage_available"
        ):
            fail("success provider usage availability does not reproduce")
        if cache_path is not None and cache_path.is_file():
            if cache_path.name != f"{cache_key}.json":
                fail("success cache filename does not match cache_key")
            try:
                cache_value = _read_object(cache_path)
            except Exception as exc:
                fail(f"success cache is unreadable: {exc}")
                cache_value = {}
            if cache_value.get("cache_key") != cache_key:
                fail("success cache payload does not reproduce cache_key")
        else:
            cache_value = {}
        if raw_path is not None and raw_path.is_file():
            try:
                raw_value = _read_object(raw_path)
            except Exception as exc:
                fail(f"success raw is unreadable: {exc}")
                raw_value = {}
            raw_payload = raw_value.get("request_payload")
            if raw_value.get("cache_key") != cache_key:
                fail("success raw payload does not reproduce cache_key")
            if (
                not isinstance(raw_payload, Mapping)
                or _mapping_sha256(raw_payload) not in payload_hashes
            ):
                fail("success raw request payload does not bind the attempts")
            if cache_value and (
                cache_value.get("raw_output_path")
                not in (str(raw_path), raw_path.relative_to(root).as_posix())
                or cache_value.get("content") != raw_value.get("content")
                or cache_value.get("provider_payload")
                != raw_value.get("provider_payload")
            ):
                fail("success cache does not reproduce the raw response")
            content = raw_value.get("content")
            if (
                not isinstance(content, str)
                or row.get("answer_response_hash")
                != hashlib.sha256(content.encode("utf-8")).hexdigest()
            ):
                fail("success result response hash does not reproduce raw content")
        try:
            case_report = audit_sensitivity_case(root, example_id, row, raw_dir)
        except Exception as exc:
            fail(f"success raw semantic audit failed: {exc}")
            case_report = failed_case
    elif terminal_outcome == "failure":
        failure_fields_empty = all(
            terminal.get(field) is None
            for field in (
                "success_raw_path",
                "success_raw_sha256",
                "success_cache_path",
                "success_cache_sha256",
            )
        )
        if not failure_fields_empty:
            fail("failure terminal contains success raw/cache evidence")
        error_category = terminal.get("terminal_error_category")
        evidence["error_category"] = error_category
        if (
            not isinstance(error_category, str)
            or last_outcome != error_category
            or row.get("error_category") != error_category
        ):
            fail("failure terminal error category does not reproduce")
        if row.get("raw_output_paths") not in ([], None):
            fail("failure result references a success raw artifact")
        if any(
            row.get(field) is not None
            for field in (
                "answer_raw_output_path",
                "answer_cache_key",
                "answer_response_hash",
            )
        ):
            fail("failure result contains success response/cache identity")
        if row.get("infrastructure_error") is not True:
            fail("failure terminal is not marked as an infrastructure error")
        if terminal.get("provider_usage_available") is not False or row.get(
            "provider_usage_available"
        ) is not False:
            fail("failed transport claims provider usage availability")
        case_report = failed_case
    else:
        fail("terminal_outcome is neither success nor failure")
        case_report = failed_case
    evidence["provider_usage_available"] = terminal.get(
        "provider_usage_available", False
    )
    return {"case_report": case_report, "evidence": evidence}


def _resolved_evidence_path(
    root: Path, value: Any, expected_parent: Path
) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    candidate = Path(value)
    candidate = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        candidate.relative_to(expected_parent.resolve())
    except ValueError:
        return None
    return candidate


def _relative_file_set(root: Path, directory: Path) -> set[str]:
    if not directory.is_dir():
        return set()
    return {
        path.relative_to(root).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
    }


def _same_number(left: Any, right: float) -> bool:
    return (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and math.isclose(float(left), right, rel_tol=1e-12, abs_tol=1e-6)
    )


def _mapping_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _expected_logical_request_id(request_tag: str, cache_key: str) -> str:
    safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", request_tag).strip("._-")[:48]
    prefix = safe_tag or "request"
    identity = hashlib.sha256(
        f"{request_tag}\0{cache_key}".encode("utf-8")
    ).hexdigest()[:16]
    return f"{prefix}-{identity}"


def criteria_from_case_reports(
    case_reports: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    """Recompute every operational criterion without accepting claimed counts."""

    return {
        "case_count": len(case_reports),
        "finish_reason_present_and_not_length_count": sum(
            item.get("finish_reason_present_and_not_length") is True
            for item in case_reports
        ),
        "unique_legal_anchored_final_label_count": sum(
            item.get("unique_legal_anchored_final_label") is True
            for item in case_reports
        ),
        "infrastructure_error_count": sum(
            item.get("infrastructure_error") is True for item in case_reports
        ),
        "hidden_reasoning_count": sum(
            item.get("hidden_reasoning") is True for item in case_reports
        ),
        "contract_failure_count": sum(
            item.get("contract_failure") is True for item in case_reports
        ),
    }


def canonical_selected_budget(
    policy: Mapping[str, Any], decision_input: Mapping[str, Any]
) -> int:
    """Select 3000/4096 from an exact, forbidden-field-free evidence schema."""

    if set(decision_input) != DECISION_INPUT_TOP_LEVEL_FIELDS:
        raise ValueError("decision input top-level fields do not match the whitelist")
    forbidden = _forbidden_fields_present(decision_input)
    if forbidden:
        raise ValueError(f"forbidden decision input fields: {sorted(forbidden)}")
    case_ids = decision_input.get("case_ids")
    reports = decision_input.get("case_reports")
    criteria = decision_input.get("criteria")
    if (
        not isinstance(case_ids, list)
        or not isinstance(reports, list)
        or not isinstance(criteria, Mapping)
        or len(case_ids) != len(set(case_ids))
        or len(case_ids) != 5
        or len(reports) != 5
    ):
        raise ValueError("decision input must contain exactly five unique cases")
    if any(
        not isinstance(item, Mapping) or set(item) != DECISION_CASE_FIELDS
        for item in reports
    ):
        raise ValueError("decision case fields do not match the whitelist")
    if set(criteria) != DECISION_CRITERIA_FIELDS:
        raise ValueError("decision criteria fields do not match the whitelist")
    if [item.get("example_id") for item in reports] != case_ids:
        raise ValueError("decision case order/identity mismatch")
    recomputed = criteria_from_case_reports(reports)
    if dict(criteria) != recomputed:
        raise ValueError("claimed criteria do not equal raw-evidence recomputation")
    expected_ids = policy.get("sensitivity", {}).get("case_ids")
    if case_ids != expected_ids:
        raise ValueError("decision cases do not equal the frozen sensitivity IDs")
    rule = policy.get("operational_rule", {})
    required = rule.get("use_4096_if_and_only_if")
    if not isinstance(required, Mapping) or set(required) != DECISION_CRITERIA_FIELDS:
        raise ValueError("authorization policy has an invalid operational rule")
    if recomputed["case_count"] != 5:
        raise ValueError("incomplete sensitivity cannot select a budget")
    return 4096 if dict(recomputed) == dict(required) else 3000


def build_budget_decision(root: Path) -> dict[str, Any]:
    """Create only the canonical budget-decision artifact, never Final active files."""

    root = root.resolve()
    policy = _read_object(root / AUTHORIZATION_POLICY_PATH)
    activation = policy["activation"]
    decision_path = root / str(activation["budget_decision_path"])
    active_config_path = root / str(activation["active_config_path"])
    active_lock_path = root / str(activation["active_lock_path"])
    if active_config_path.exists() or active_lock_path.exists():
        raise FileExistsError(
            "budget selection requires inactive Final config and lock"
        )
    if decision_path.exists():
        raise FileExistsError("refusing to overwrite an existing budget decision")
    sensitivity = audit_sensitivity(root, preflight_only=False)
    if (
        sensitivity.get("status") != "complete"
        or sensitivity.get("decision_authorized") is not True
        or not isinstance(sensitivity.get("canonical_decision"), Mapping)
    ):
        raise ValueError(
            "complete five-case sensitivity evidence with decision_authorized=true "
            "is required"
        )
    audit_path = root / str(policy["sensitivity"]["audit_path"])
    if (
        not audit_path.is_file()
        or audit_path.read_bytes() != canonical_json_bytes(sensitivity)
    ):
        raise ValueError("sensitivity audit artifact does not exactly reproduce")
    decision_input = {
        "case_ids": sensitivity["frozen_case_ids"],
        "case_reports": sensitivity["case_reports"],
        "criteria": sensitivity["criteria"],
    }
    selected = canonical_selected_budget(policy, decision_input)
    if sensitivity["canonical_decision"].get("selected_max_tokens") != selected:
        raise ValueError("canonical sensitivity decision does not reproduce")
    final_candidate = _validated_candidate_spec(
        root, policy["final_candidates"], selected, "Final"
    )
    contract_candidate = _validated_candidate_spec(
        root, policy["contract_candidates"], selected, "Contract"
    )
    function_spec = policy["decision_function"]
    function_path = root / str(function_spec["path"])
    if (
        function_spec.get("name") != DECISION_FUNCTION_NAME
        or function_spec.get("version") != DECISION_FUNCTION_VERSION
        or function_spec.get("sha256") != file_sha256(function_path)
    ):
        raise ValueError("canonical decision function identity changed")
    artifact = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "artifact_type": "canonical_final_budget_decision",
        "status": "complete",
        "sensitivity_evidence": {
            "config": {
                "path": SENSITIVITY_CONFIG_PATH,
                "sha256": file_sha256(root / SENSITIVITY_CONFIG_PATH),
            },
            "sample_manifest": {
                "path": SENSITIVITY_MANIFEST_PATH,
                "sha256": file_sha256(root / SENSITIVITY_MANIFEST_PATH),
            },
            "policy": {
                "path": AUTHORIZATION_POLICY_PATH,
                "sha256": file_sha256(root / AUTHORIZATION_POLICY_PATH),
            },
            "candidate_contract": {
                "path": CANDIDATE_CONTRACT_PATH,
                "sha256": file_sha256(root / CANDIDATE_CONTRACT_PATH),
            },
            "results": {
                "path": sensitivity["result_path"],
                "sha256": sensitivity["result_sha256"],
            },
            "run_manifest": {
                "path": sensitivity["run_manifest_path"],
                "sha256": sensitivity["run_manifest_sha256"],
            },
            "raw_tree": sensitivity["raw_tree"],
            "cache_tree": sensitivity["cache_tree"],
            "attempt_tree": sensitivity["attempt_tree"],
            "terminal_tree": sensitivity["terminal_tree"],
            "runtime_profile": {
                "path": RUNTIME_PROFILE_PATH,
                "sha256": file_sha256(root / RUNTIME_PROFILE_PATH),
            },
            "audit": {
                "path": audit_path.relative_to(root).as_posix(),
                "sha256": file_sha256(audit_path),
            },
        },
        "operational_criteria": sensitivity["criteria"],
        "decision_function": dict(function_spec),
        "decision_input_fields": sensitivity["decision_input_fields"],
        "forbidden_fields_present": sensitivity["forbidden_fields_present"],
        "decision_recomputed_from_raw_evidence": sensitivity[
            "decision_recomputed_from_raw_evidence"
        ],
        "gold_evaluation_disabled": sensitivity["gold_evaluation_disabled"],
        "selected_max_tokens": selected,
        "selected_final_candidate": final_candidate,
        "selected_contract_candidate": contract_candidate,
        "model_calls_made": 0,
        "network_calls_made": 0,
        "decision_authorized": True,
    }
    if set(artifact) != BUDGET_DECISION_FIELDS:
        raise AssertionError(
            "budget decision implementation drifted from its whitelist"
        )
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    with decision_path.open("xb") as handle:
        handle.write(canonical_json_bytes(artifact))
    return artifact


def audit_budget_decision(root: Path) -> dict[str, Any]:
    """Recompute sensitivity and reject any inconsistent decision artifact."""

    root = root.resolve()
    policy = _read_object(root / AUTHORIZATION_POLICY_PATH)
    decision_path = root / str(policy["activation"]["budget_decision_path"])
    if not decision_path.is_file():
        return {
            "status": "pending",
            "blocking": ["canonical budget decision is missing"],
            "selected_max_tokens": None,
        }
    try:
        decision = _read_object(decision_path)
    except Exception as exc:
        return {"status": "fail", "blocking": [str(exc)], "selected_max_tokens": None}
    sensitivity = audit_sensitivity(
        root, preflight_only=False, allow_downstream_artifacts=True
    )
    decision_forbidden = sorted(_forbidden_fields_present(decision))
    checks: dict[str, bool] = {
        "decision_payload_whitelist": set(decision) == BUDGET_DECISION_FIELDS,
        "decision_schema": decision.get("schema_version") == 1
        and decision.get("protocol") == PROTOCOL
        and decision.get("artifact_type") == "canonical_final_budget_decision"
        and decision.get("status") == "complete"
        and decision.get("decision_authorized") is True,
        "sensitivity_complete": sensitivity.get("status") == "complete"
        and sensitivity.get("decision_authorized") is True,
        "gold_evaluation_disabled": sensitivity.get("gold_evaluation_disabled") is True
        and decision.get("gold_evaluation_disabled") is True,
        "forbidden_fields_absent": sensitivity.get("forbidden_fields_present") == []
        and decision.get("forbidden_fields_present") == []
        and decision_forbidden == [],
    }
    selected: int | None = None
    if sensitivity.get("status") == "complete":
        decision_input = {
            "case_ids": sensitivity["frozen_case_ids"],
            "case_reports": sensitivity["case_reports"],
            "criteria": sensitivity["criteria"],
        }
        try:
            selected = canonical_selected_budget(policy, decision_input)
        except Exception:
            selected = None
    checks["criteria_exactly_recomputed"] = (
        selected is not None
        and decision.get("operational_criteria") == sensitivity.get("criteria")
        and decision.get("selected_max_tokens") == selected
        and decision.get("decision_recomputed_from_raw_evidence") is True
    )
    if selected is not None:
        expected_final = _validated_candidate_spec(
            root, policy["final_candidates"], selected, "Final"
        )
        expected_contract = _validated_candidate_spec(
            root, policy["contract_candidates"], selected, "Contract"
        )
    else:
        expected_final = None
        expected_contract = None
    checks["selected_candidates_exact"] = (
        decision.get("selected_final_candidate") == expected_final
        and decision.get("selected_contract_candidate") == expected_contract
    )
    function_spec = policy.get("decision_function", {})
    function_path = root / str(function_spec.get("path", ""))
    checks["decision_function_identity"] = (
        decision.get("decision_function") == function_spec
        and function_path.is_file()
        and function_spec.get("sha256") == file_sha256(function_path)
    )
    audit_path = root / str(policy["sensitivity"]["audit_path"])
    expected_sensitivity_evidence: dict[str, Any] | None = None
    if sensitivity.get("status") == "complete" and audit_path.is_file():
        expected_sensitivity_evidence = {
            "config": {
                "path": SENSITIVITY_CONFIG_PATH,
                "sha256": file_sha256(root / SENSITIVITY_CONFIG_PATH),
            },
            "sample_manifest": {
                "path": SENSITIVITY_MANIFEST_PATH,
                "sha256": file_sha256(root / SENSITIVITY_MANIFEST_PATH),
            },
            "policy": {
                "path": AUTHORIZATION_POLICY_PATH,
                "sha256": file_sha256(root / AUTHORIZATION_POLICY_PATH),
            },
            "candidate_contract": {
                "path": CANDIDATE_CONTRACT_PATH,
                "sha256": file_sha256(root / CANDIDATE_CONTRACT_PATH),
            },
            "results": {
                "path": sensitivity["result_path"],
                "sha256": sensitivity["result_sha256"],
            },
            "run_manifest": {
                "path": sensitivity["run_manifest_path"],
                "sha256": sensitivity["run_manifest_sha256"],
            },
            "raw_tree": sensitivity["raw_tree"],
            "cache_tree": sensitivity["cache_tree"],
            "attempt_tree": sensitivity["attempt_tree"],
            "terminal_tree": sensitivity["terminal_tree"],
            "runtime_profile": {
                "path": RUNTIME_PROFILE_PATH,
                "sha256": file_sha256(root / RUNTIME_PROFILE_PATH),
            },
            "audit": {
                "path": audit_path.relative_to(root).as_posix(),
                "sha256": file_sha256(audit_path),
            },
        }
    checks["sensitivity_evidence_identity"] = (
        expected_sensitivity_evidence is not None
        and decision.get("sensitivity_evidence") == expected_sensitivity_evidence
    )
    checks["decision_audit_metadata_exact"] = (
        decision.get("decision_input_fields")
        == sensitivity.get("decision_input_fields")
        and decision.get("model_calls_made") == 0
        and decision.get("network_calls_made") == 0
    )
    checks["sensitivity_audit_exact"] = (
        audit_path.is_file()
        and sensitivity.get("status") == "complete"
        and audit_path.read_bytes() == canonical_json_bytes(sensitivity)
        and decision.get("sensitivity_evidence", {}).get("audit", {}).get("sha256")
        == file_sha256(audit_path)
    )
    checks["decision_bytes_canonical"] = (
        decision_path.read_bytes() == canonical_json_bytes(decision)
    )
    blocking = [name for name, passed in checks.items() if not passed]
    return {
        "status": "pass" if not blocking else "fail",
        "checks": checks,
        "blocking": blocking,
        "selected_max_tokens": selected,
        "decision_path": decision_path.relative_to(root).as_posix(),
        "decision_sha256": file_sha256(decision_path),
        "decision": decision,
        "sensitivity": sensitivity,
        "forbidden_fields_present": decision_forbidden,
    }


def verify_sensitivity_runtime(
    root: Path, *, config_hash: str, config: Any
) -> Mapping[str, Any]:
    """Python-side sensitivity arm called before ``create_client``."""

    if os.environ.get(SENSITIVITY_ARM_ENV) != SENSITIVITY_ARM_VALUE:
        raise DatasetSchemaError(
            "sensitivity is not armed; require "
            f"{SENSITIVITY_ARM_ENV}={SENSITIVITY_ARM_VALUE}"
        )
    expected_hash = file_sha256(root / SENSITIVITY_CONFIG_PATH)
    if config_hash != expected_hash:
        raise DatasetSchemaError("sensitivity config hash is not the frozen config")
    if (
        getattr(config, "protocol_version", None) != PROTOCOL
        or getattr(config, "attempt_id", None) != 2
        or getattr(config, "execution_revision", None)
        != "attempt2-infrastructure-v1"
        or getattr(config, "sample_set", None) != "cot4096_length_5"
        or tuple(getattr(config, "methods", ())) != ("cot",)
        or getattr(getattr(config, "client", None), "max_tokens", None) != 4096
        or getattr(getattr(config, "client", None), "timeout_seconds", None)
        != 360
        or getattr(getattr(config, "client", None), "max_retries", None) != 1
        or getattr(
            getattr(config, "client", None), "retry_backoff_seconds", None
        )
        != 1.0
        or getattr(
            getattr(config, "client", None), "fresh_cache_required", None
        )
        is not True
        or getattr(config, "gold_evaluation", None) is not False
    ):
        raise DatasetSchemaError("sensitivity config semantics are not frozen")
    report = audit_sensitivity(
        root, preflight_only=True, require_runtime_arm=True
    )
    if report.get("status") != "pass" or not report.get(
        "execution_ready_after_attempt2_commit"
    ):
        raise DatasetSchemaError(
            f"sensitivity Python preflight failed closed: {report.get('blocking')}"
        )
    return report


def verify_contract_candidate_runtime(
    root: Path, *, config_hash: str, config: Any
) -> Mapping[str, Any]:
    """Bind Contract inference to the selected budget and current code."""

    if os.environ.get(CONTRACT_ARM_ENV) != CONTRACT_ARM_VALUE:
        raise DatasetSchemaError(
            f"Contract-12 is not armed; require {CONTRACT_ARM_ENV}={CONTRACT_ARM_VALUE}"
        )
    decision = audit_budget_decision(root)
    if decision.get("status") != "pass":
        raise DatasetSchemaError(
            "Contract-12 requires a passing budget decision: "
            f"{decision.get('blocking')}"
        )
    selected = int(decision["selected_max_tokens"])
    selected_spec = decision["decision"]["selected_contract_candidate"]
    if (
        config_hash != selected_spec.get("sha256")
        or getattr(getattr(config, "client", None), "max_tokens", None) != selected
        or getattr(config, "protocol_version", None) != PROTOCOL
        or getattr(config, "sample_set", None) != "contract_12"
    ):
        raise DatasetSchemaError(
            "Contract candidate does not match selected_max_tokens"
        )
    binding = audit_contract_candidate_binding(root, config_hash, selected)
    if binding["status"] != "pass":
        raise DatasetSchemaError(
            f"current Contract candidate lock failed: {binding['blocking']}"
        )
    paths = (
        root / str(getattr(config, "output_path")),
        root / str(getattr(getattr(config, "client"), "raw_dir")),
        root / str(getattr(getattr(config, "client"), "cache_dir")),
        root / str(getattr(getattr(config, "client"), "attempts_dir")),
        root / str(getattr(getattr(config, "client"), "terminal_dir")),
    )
    inferred = paths[0].parent
    paths = (
        *paths,
        inferred / "contract_audit.json",
        inferred / "g1_manual_review.json",
    )
    existing = [path.relative_to(root).as_posix() for path in paths if path.exists()]
    if existing:
        raise DatasetSchemaError(f"Contract candidate paths are not fresh: {existing}")
    return {"decision": decision, "binding": binding}


def audit_contract_candidate_binding(
    root: Path, config_hash: str, selected_budget: int
) -> dict[str, Any]:
    """Validate the shared current-code Contract lock without requiring inference."""

    lock_path = root / CONTRACT_CANDIDATE_LOCK_PATH
    if not lock_path.is_file():
        return {"status": "fail", "blocking": ["Contract candidate lock is missing"]}
    lock = _read_object(lock_path)
    expected_configs = lock.get("candidate_config_sha256", {})
    groups = (
        "artifact_sha256",
        "prompt_sha256",
        "inference_contract_sha256",
        "tooling_sha256",
    )
    checks = {
        "schema_protocol": lock.get("schema_version") == 2
        and lock.get("protocol") == PROTOCOL
        and lock.get("artifact_type") == "current_code_contract_candidate_lock"
        and lock.get("attempt_id") == 2
        and lock.get("execution_revision") == "attempt2-infrastructure-v1",
        "selected_config_hash": expected_configs.get(str(selected_budget))
        == config_hash,
        "both_candidate_hashes": all(
            expected_configs.get(str(budget))
            == file_sha256(root / path)
            for budget, path in CONTRACT_CANDIDATE_PATHS.items()
        ),
        "dataset_hash": lock.get("dataset_sha256")
        == "1a2e13c9e57c535da46c22622cc46185fd1a13ce49522660eda1eed38509db98",
        "runtime_profile": lock.get("runtime_profile")
        == {
            "path": RUNTIME_PROFILE_PATH,
            "sha256": file_sha256(root / RUNTIME_PROFILE_PATH),
        },
    }
    for group in groups:
        expected = lock.get(group)
        checks[f"{group}_current"] = (
            isinstance(expected, Mapping)
            and bool(expected)
            and all(
                _locked_file_matches(root, relative, digest)
                for relative, digest in expected.items()
            )
        )
    blocking = [name for name, passed in checks.items() if not passed]
    return {
        "status": "pass" if not blocking else "fail",
        "checks": checks,
        "blocking": blocking,
        "lock_path": CONTRACT_CANDIDATE_LOCK_PATH,
        "lock_sha256": file_sha256(lock_path),
        "inference_contract_sha256": lock.get("inference_contract_sha256", {}),
    }


def audit_contract_candidate_freeze(
    root: Path, *, require_fresh: bool = False
) -> dict[str, Any]:
    """Check that the two Contract candidates differ only by budget paths."""

    raw = {
        budget: _read_object(root / path)
        for budget, path in CONTRACT_CANDIDATE_PATHS.items()
    }
    differences = _mapping_differences(raw[3000], raw[4096])
    allowed = {
        "client.attempts_dir",
        "client.cache_dir",
        "client.max_tokens",
        "client.raw_dir",
        "client.terminal_dir",
        "output_path",
    }
    methods = [
        "direct",
        "cot",
        "cot_refine",
        "constrained",
        "gate_cot",
        "repair_only",
        "vgcf2",
    ]
    checks = {
        "only_budget_and_output_paths_differ": differences == allowed,
        "same_contract_examples": all(
            item.get("sample_manifest_path") == "configs/v2_3_contract_samples.json"
            and item.get("sample_set") == "contract_12"
            and item.get("sample_count") == 12
            for item in raw.values()
        ),
        "same_seven_methods": all(
            item.get("methods") == methods for item in raw.values()
        ),
        "same_prompt_model_transport": all(
            raw[3000].get(field) == raw[4096].get(field)
            for field in (
                "dataset_path",
                "expected_dataset_sha256",
                "prompt_directory",
                "protocol_lock_path",
                "gold_evaluation",
                "seed",
                "sampling_strategy",
                "max_questions_per_theory",
                "attempt_id",
                "execution_revision",
                "runtime_profile_path",
                "runtime_profile_sha256",
            )
        ),
        "budgets_exact": raw[3000].get("client", {}).get("max_tokens") == 3000
        and raw[4096].get("client", {}).get("max_tokens") == 4096,
        "shared_lock": raw[3000].get("protocol_lock_path")
        == raw[4096].get("protocol_lock_path")
        == CONTRACT_CANDIDATE_LOCK_PATH,
        "shared_attempt2_runtime_and_retry": all(
            item.get("attempt_id") == 2
            and item.get("execution_revision") == "attempt2-infrastructure-v1"
            and item.get("runtime_profile_path") == RUNTIME_PROFILE_PATH
            and item.get("runtime_profile_sha256")
            == file_sha256(root / RUNTIME_PROFILE_PATH)
            and item.get("client", {}).get("timeout_seconds") == 360
            and item.get("client", {}).get("max_retries") == 1
            and item.get("client", {}).get("retry_backoff_seconds") == 1.0
            and item.get("client", {}).get("fresh_cache_required") is True
            for item in raw.values()
        ),
    }
    # Python has no set-comprehension flattening syntax; compute the six paths
    # separately so this check remains easy to audit.
    candidate_paths = {
        raw[budget][field]
        for budget in (3000, 4096)
        for field in ("output_path",)
    } | {
        raw[budget]["client"][field]
        for budget in (3000, 4096)
        for field in ("raw_dir", "cache_dir", "attempts_dir", "terminal_dir")
    }
    if require_fresh:
        checks["fresh_paths"] = len(candidate_paths) == 10 and not any(
            (root / relative).exists() for relative in candidate_paths
        )
    bindings = {
        budget: audit_contract_candidate_binding(
            root, file_sha256(root / path), budget
        )
        for budget, path in CONTRACT_CANDIDATE_PATHS.items()
    }
    checks["current_code_lock"] = all(
        report["status"] == "pass" for report in bindings.values()
    )
    blocking = [name for name, passed in checks.items() if not passed]
    return {
        "status": "pass" if not blocking else "fail",
        "checks": checks,
        "blocking": blocking,
        "differences": sorted(differences),
        "candidate_sha256": {
            str(budget): file_sha256(root / path)
            for budget, path in CONTRACT_CANDIDATE_PATHS.items()
        },
        "bindings": bindings,
    }


def audit_primary_hypotheses(root: Path) -> dict[str, Any]:
    """Validate the immutable H1/H2 preregistration mechanically."""

    path = root / PRIMARY_HYPOTHESES_PATH
    policy = _read_object(root / AUTHORIZATION_POLICY_PATH)
    value = _read_object(path)
    family = value.get("primary_family", {})
    hypotheses = family.get("hypotheses", [])
    pairs = [
        (item.get("id"), item.get("left_method"), item.get("right_method"))
        for item in hypotheses
        if isinstance(item, Mapping)
    ]
    checks = {
        "frozen_hash": policy.get("frozen_artifact_sha256", {}).get(
            PRIMARY_HYPOTHESES_PATH
        )
        == file_sha256(path),
        "schema_protocol": value.get("schema_version") == 1
        and value.get("protocol") == PROTOCOL
        and value.get("freeze_status")
        == "preregistered_before_final_results",
        "manifest_binding": value.get("final_sample", {}).get("manifest_path")
        == FINAL_MANIFEST_PATH
        and value.get("final_sample", {}).get("manifest_sha256")
        == file_sha256(root / FINAL_MANIFEST_PATH)
        and value.get("final_sample", {}).get("paired_n") == 300,
        "exact_h1_h2": pairs
        == [
            ("H1", "vgcf2", "cot_refine"),
            ("H2", "vgcf2", "gate_cot"),
        ],
        "two_sided_exact_mcnemar": len(hypotheses) == 2
        and all(
            isinstance(item, Mapping)
            and item.get("test") == "two_sided_exact_mcnemar"
            for item in hypotheses
        ),
        "holm_family_two": family.get("family_size") == 2
        and family.get("multiplicity", {}).get("scope") == "H1 and H2 only"
        and str(family.get("multiplicity", {}).get("method", "")).startswith(
            "Holm"
        ),
        "paired_ci_required": family.get("paired_confidence_interval", {}).get(
            "sampling_unit"
        )
        == "paired example_id"
        and "paired_ci_95" in family.get("required_report_fields", []),
        "immutable": value.get("immutability", {}).get("amendment_allowed")
        is False,
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "path": PRIMARY_HYPOTHESES_PATH,
        "sha256": file_sha256(path),
    }


def audit_protected_artifacts(root: Path) -> dict[str, Any]:
    """Verify historical Dev/Confirmation/Contract and quarantine bytes."""

    baseline_path = root / PROTECTED_ARTIFACTS_PATH
    baseline = _read_object(baseline_path)
    tree_matches: dict[str, bool] = {}
    actual_trees: dict[str, dict[str, Any]] = {}
    for relative, expected in baseline.get("protected_trees", {}).items():
        actual = tree_snapshot(root / relative)
        comparable = {
            key: actual[key]
            for key in ("file_count", "total_bytes", "tree_sha256")
        }
        actual_trees[relative] = comparable
        tree_matches[relative] = comparable == expected
    file_hashes = {
        relative: file_sha256(root / relative)
        for relative in baseline.get("protected_files", {})
        if (root / relative).is_file()
    }
    file_matches = {
        relative: file_hashes.get(relative) == expected
        for relative, expected in baseline.get("protected_files", {}).items()
    }
    checks = {
        "baseline_schema": baseline.get("schema_version") == 1
        and baseline.get("protocol") == PROTOCOL,
        "trees": bool(tree_matches) and all(tree_matches.values()),
        "files": bool(file_matches) and all(file_matches.values()),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "tree_matches": tree_matches,
        "file_matches": file_matches,
        "actual_trees": actual_trees,
        "actual_files": file_hashes,
        "path": PROTECTED_ARTIFACTS_PATH,
        "sha256": file_sha256(baseline_path),
    }


def audit_final_manifest_reproduction(root: Path) -> dict[str, Any]:
    """Rebuild Final-300 from meta-test and both quarantines."""

    from .data import load_examples
    from .final_v2_3_1 import audit_final_manifest
    from .protocol import (
        identify_dataset,
        load_quarantine,
        load_sample_manifest,
        load_theory_quarantine,
    )

    dataset = root / (
        "data/proofwriter/proofwriter-dataset-V2020.12.3/OWA/depth-5/"
        "meta-test.jsonl"
    )
    question_path = root / "configs/quarantine_ids.json"
    theory_path = root / "configs/quarantine_theories.json"
    manifest_path = root / FINAL_MANIFEST_PATH
    identity = identify_dataset(dataset)
    questions = load_quarantine(question_path)
    theories = load_theory_quarantine(theory_path)
    manifest = load_sample_manifest(manifest_path)
    report = audit_final_manifest(
        manifest,
        load_examples(dataset),
        dataset_sha256=identity.sha256,
        question_quarantine=questions,
        theory_quarantine_ids=theories.theory_ids,
        question_quarantine_path=question_path,
        theory_quarantine_path=theory_path,
    )
    report = dict(report)
    report["path"] = FINAL_MANIFEST_PATH
    report["sha256"] = file_sha256(manifest_path)
    return report


def validate_final_prerequisites(
    root: Path, *, require_active: bool
) -> dict[str, Any]:
    """Recompute A-E of Final authorization from source artifacts."""

    from .contract_v2_3_1 import (
        audit_current_contract,
        contract_audit_bytes,
        selected_contract_paths,
    )

    root = root.resolve()
    decision = audit_budget_decision(root)
    hash_contract = audit_hash_contract(root)
    protected = audit_protected_artifacts(root)
    manifest = audit_final_manifest_reproduction(root)
    hypotheses = audit_primary_hypotheses(root)
    checks = {
        "sensitivity_and_budget_decision": decision.get("status") == "pass",
        "candidate_contract_current": hash_contract.get("status") == "pass",
        "protected_artifacts": protected.get("status") == "pass",
        "final_manifest_exact_reproduction": manifest.get("status") == "pass",
        "primary_hypotheses": hypotheses.get("status") == "pass",
    }
    selected = decision.get("selected_max_tokens")
    final_candidate: dict[str, Any] | None = None
    candidate_value: dict[str, Any] = {}
    final_paths: dict[str, str] = {}
    if selected in (3000, 4096):
        policy = _read_object(root / AUTHORIZATION_POLICY_PATH)
        final_candidate = _validated_candidate_spec(
            root, policy["final_candidates"], int(selected), "Final"
        )
        candidate_path = root / final_candidate["path"]
        candidate_value = _read_object(candidate_path)
        client = candidate_value.get("client", {})
        final_paths = {
            "output": str(candidate_value.get("output_path")),
            "raw": str(client.get("raw_dir")),
            "cache": str(client.get("cache_dir")),
            "attempts": str(client.get("attempts_dir")),
            "terminal": str(client.get("terminal_dir")),
        }
        checks["selected_candidate_budget"] = client.get("max_tokens") == selected
        checks["selected_candidate_matches_decision"] = (
            decision.get("decision", {}).get("selected_final_candidate")
            == final_candidate
        )
        checks["final_paths_fresh"] = all(
            relative and not (root / relative).exists()
            for relative in final_paths.values()
        )
    else:
        checks["selected_candidate_budget"] = False
        checks["selected_candidate_matches_decision"] = False
        checks["final_paths_fresh"] = False

    active_config_path = root / ACTIVE_FINAL_CONFIG_PATH
    active_lock_path = root / ACTIVE_FINAL_LOCK_PATH
    if require_active and final_candidate is not None:
        checks["active_config_exact_candidate_bytes"] = active_config_exact_candidate(
            root, final_candidate
        )
        checks["active_lock_present"] = active_lock_path.is_file()
    else:
        checks["active_config_and_lock_absent"] = (
            not active_config_path.exists() and not active_lock_path.exists()
        )

    contract = audit_current_contract(root, require_results=True)
    stored_contract_audit: dict[str, Any] = {}
    if selected in (3000, 4096):
        contract_paths = selected_contract_paths(root, int(selected))
        stored_path = contract_paths["audit"]
        stored_contract_audit = {
            "path": stored_path.relative_to(root).as_posix(),
            "sha256": file_sha256(stored_path) if stored_path.is_file() else None,
            "canonical_match": stored_path.is_file()
            and stored_path.read_bytes() == contract_audit_bytes(contract),
        }
    checks["current_code_current_budget_contract_pass"] = (
        contract.get("status") == "pass"
        and contract.get("budget_decision", {}).get("selected_max_tokens")
        == selected
        and contract.get("infrastructure_error_count") == 0
        and contract.get("hidden_reasoning_count") == 0
        and contract.get("shared_response_audit", {}).get("status") == "pass"
        and contract.get("g1", {}).get("status") == "pass"
        and contract.get("g1", {}).get("manual_review", {}).get("reviewed_count")
        == 12
        and contract.get("g1", {}).get("b_c_mismatch_count") == 0
    )
    checks["contract_audit_canonical_artifact"] = stored_contract_audit.get(
        "canonical_match"
    ) is True
    blocking = [name for name, passed in checks.items() if not passed]
    return {
        "status": "pass" if not blocking else "fail",
        "checks": checks,
        "blocking": blocking,
        "selected_max_tokens": selected,
        "selected_final_candidate": final_candidate,
        "final_candidate_config": candidate_value,
        "final_paths": final_paths,
        "budget_decision": decision,
        "candidate_contract": hash_contract,
        "protected_artifacts": protected,
        "final_manifest": manifest,
        "primary_hypotheses": hypotheses,
        "contract": contract,
        "stored_contract_audit": stored_contract_audit,
    }


def active_config_exact_candidate(
    root: Path, candidate: Mapping[str, Any]
) -> bool:
    """Return true only for a byte-for-byte selected-candidate copy."""

    try:
        candidate_path = _resolve_relative_file(
            root.resolve(), candidate.get("path"), "selected Final candidate"
        )
    except (DatasetSchemaError, TypeError, ValueError):
        return False
    active = root.resolve() / ACTIVE_FINAL_CONFIG_PATH
    return (
        active.is_file()
        and active.read_bytes() == candidate_path.read_bytes()
        and candidate.get("sha256") == file_sha256(candidate_path)
        and file_sha256(active) == candidate.get("sha256")
    )


def audit_final_authorization_status(root: Path) -> dict[str, Any]:
    """Report a truthful pending state before sensitivity/Contract evidence exists."""

    root = root.resolve()
    static = {
        "candidate_contract": audit_hash_contract(root),
        "contract_candidates": audit_contract_candidate_freeze(
            root, require_fresh=True
        ),
        "protected_artifacts": audit_protected_artifacts(root),
        "final_manifest": audit_final_manifest_reproduction(root),
        "primary_hypotheses": audit_primary_hypotheses(root),
    }
    static_pass = all(item.get("status") == "pass" for item in static.values())
    decision = audit_budget_decision(root)
    if decision.get("status") == "pending":
        return {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "status": "pending" if static_pass else "fail",
            "run_status": "not_run",
            "blocking": list(decision.get("blocking", [])),
            "static": static,
            "budget_decision": decision,
            "active_final_config_exists": (root / ACTIVE_FINAL_CONFIG_PATH).exists(),
            "active_final_lock_exists": (root / ACTIVE_FINAL_LOCK_PATH).exists(),
            "model_calls_made": 0,
            "network_calls_made": 0,
        }
    prerequisites = validate_final_prerequisites(root, require_active=False)
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": prerequisites["status"],
        "run_status": "authorization_ready"
        if prerequisites["status"] == "pass"
        else "blocked",
        "blocking": prerequisites["blocking"],
        "static": static,
        "prerequisites": prerequisites,
        "active_final_config_exists": (root / ACTIVE_FINAL_CONFIG_PATH).exists(),
        "active_final_lock_exists": (root / ACTIVE_FINAL_LOCK_PATH).exists(),
        "model_calls_made": 0,
        "network_calls_made": 0,
    }


FINAL_LOCK_FIELDS = frozenset(
    {
        "schema_version",
        "protocol",
        "artifact_type",
        "status",
        "arm_environment_variable",
        "arm_value",
        "selected_max_tokens",
        "config_sha256",
        "selected_candidate",
        "authorization_policy",
        "candidate_contract",
        "runtime_profile",
        "budget_decision",
        "sensitivity_evidence",
        "contract_authorization",
        "dataset_sha256",
        "quarantine_sha256",
        "sample_manifest",
        "primary_hypotheses",
        "prompt_sha256",
        "inference_contract_sha256",
        "final_paths",
        "prohibited_next_budget",
    }
)


def authorize_final(root: Path) -> dict[str, Any]:
    """Exclusively create active Final config and schema-5 lock after A-E pass."""

    root = root.resolve()
    active_config_path = root / ACTIVE_FINAL_CONFIG_PATH
    active_lock_path = root / ACTIVE_FINAL_LOCK_PATH
    if active_config_path.exists() or active_lock_path.exists():
        raise FileExistsError("refusing to overwrite active Final config or lock")
    prerequisites = validate_final_prerequisites(root, require_active=False)
    if prerequisites["status"] != "pass":
        raise ValueError(
            f"Final authorization prerequisites failed: {prerequisites['blocking']}"
        )
    selected = int(prerequisites["selected_max_tokens"])
    candidate = prerequisites["selected_final_candidate"]
    candidate_path = root / candidate["path"]
    candidate_value = prerequisites["final_candidate_config"]
    policy_path = root / AUTHORIZATION_POLICY_PATH
    contract_path = root / CANDIDATE_CONTRACT_PATH
    decision = prerequisites["budget_decision"]
    decision_value = decision["decision"]
    contract_report = prerequisites["contract"]
    contract_audit = prerequisites["stored_contract_audit"]
    hash_contract = _read_object(contract_path)
    lock = {
        "schema_version": 5,
        "protocol": PROTOCOL,
        "artifact_type": "semantic_final_authorization_lock",
        "status": "authorized_after_sensitivity_contract_g1",
        "arm_environment_variable": FINAL_ARM_ENV,
        "arm_value": FINAL_ARM_VALUE,
        "selected_max_tokens": selected,
        "config_sha256": candidate["sha256"],
        "selected_candidate": candidate,
        "authorization_policy": {
            "path": AUTHORIZATION_POLICY_PATH,
            "sha256": file_sha256(policy_path),
        },
        "candidate_contract": {
            "path": CANDIDATE_CONTRACT_PATH,
            "sha256": file_sha256(contract_path),
        },
        "runtime_profile": {
            "path": RUNTIME_PROFILE_PATH,
            "sha256": file_sha256(root / RUNTIME_PROFILE_PATH),
        },
        "budget_decision": {
            "path": decision["decision_path"],
            "sha256": decision["decision_sha256"],
        },
        "sensitivity_evidence": decision_value["sensitivity_evidence"],
        "contract_authorization": {
            "audit": {
                "path": contract_audit["path"],
                "sha256": contract_audit["sha256"],
            },
            "artifact_identity": contract_report["artifact_identity"],
            "status": contract_report["status"],
            "reviewed_count": contract_report["g1"]["manual_review"][
                "reviewed_count"
            ],
            "b_c_mismatch_count": contract_report["g1"][
                "b_c_mismatch_count"
            ],
        },
        "dataset_sha256": candidate_value["expected_dataset_sha256"],
        "quarantine_sha256": {
            "questions": file_sha256(root / candidate_value["quarantine_path"]),
            "theories": file_sha256(
                root / candidate_value["theory_quarantine_path"]
            ),
        },
        "sample_manifest": {
            "path": candidate_value["sample_manifest_path"],
            "sha256": file_sha256(root / candidate_value["sample_manifest_path"]),
        },
        "primary_hypotheses": {
            "path": PRIMARY_HYPOTHESES_PATH,
            "sha256": file_sha256(root / PRIMARY_HYPOTHESES_PATH),
        },
        "prompt_sha256": {
            Path(relative).name: digest
            for relative, digest in hash_contract["prompt_sha256"].items()
        },
        "inference_contract_sha256": dict(
            hash_contract["inference_contract_sha256"]
        ),
        "final_paths": prerequisites["final_paths"],
        "prohibited_next_budget": 6000,
    }
    if set(lock) != FINAL_LOCK_FIELDS:
        raise AssertionError("schema-5 lock fields drifted from their whitelist")
    active_config_path.parent.mkdir(parents=True, exist_ok=True)
    with active_config_path.open("xb") as handle:
        handle.write(candidate_path.read_bytes())
    try:
        with active_lock_path.open("xb") as handle:
            handle.write(canonical_json_bytes(lock))
        previous = os.environ.get(FINAL_ARM_ENV)
        os.environ[FINAL_ARM_ENV] = FINAL_ARM_VALUE
        try:
            validate_final_runtime(root, config_hash=candidate["sha256"], lock=lock)
        finally:
            if previous is None:
                os.environ.pop(FINAL_ARM_ENV, None)
            else:
                os.environ[FINAL_ARM_ENV] = previous
    except Exception:
        active_lock_path.unlink(missing_ok=True)
        active_config_path.unlink(missing_ok=True)
        raise
    return {
        "status": "pass",
        "selected_max_tokens": selected,
        "active_config_path": ACTIVE_FINAL_CONFIG_PATH,
        "active_config_sha256": file_sha256(active_config_path),
        "active_lock_path": ACTIVE_FINAL_LOCK_PATH,
        "active_lock_sha256": file_sha256(active_lock_path),
        "model_calls_made": 0,
        "network_calls_made": 0,
    }


def validate_final_runtime(
    root: Path, *, config_hash: str, lock: Mapping[str, Any]
) -> dict[str, Any]:
    """Schema-5 runtime semantic replay performed before ``create_client``."""

    root = root.resolve()
    checks = {
        "lock_field_whitelist": set(lock) == FINAL_LOCK_FIELDS,
        "lock_schema_protocol": lock.get("schema_version") == 5
        and lock.get("protocol") == PROTOCOL
        and lock.get("artifact_type") == "semantic_final_authorization_lock"
        and lock.get("status")
        == "authorized_after_sensitivity_contract_g1"
        and lock.get("prohibited_next_budget") == 6000,
        "runtime_arm": os.environ.get(FINAL_ARM_ENV) == FINAL_ARM_VALUE
        and lock.get("arm_environment_variable") == FINAL_ARM_ENV
        and lock.get("arm_value") == FINAL_ARM_VALUE,
        "active_lock_canonical_bytes": (root / ACTIVE_FINAL_LOCK_PATH).is_file()
        and (root / ACTIVE_FINAL_LOCK_PATH).read_bytes()
        == canonical_json_bytes(dict(lock)),
    }
    prerequisites = validate_final_prerequisites(root, require_active=True)
    if prerequisites["status"] != "pass":
        raise DatasetSchemaError(
            "schema-5 Final semantic prerequisites failed: "
            f"{prerequisites['blocking']}"
        )
    selected = prerequisites.get("selected_max_tokens")
    candidate = prerequisites.get("selected_final_candidate")
    checks["semantic_prerequisites_recomputed"] = prerequisites["status"] == "pass"
    checks["selected_budget_everywhere"] = (
        selected in (3000, 4096)
        and lock.get("selected_max_tokens") == selected
        and isinstance(candidate, Mapping)
        and lock.get("selected_candidate") == candidate
        and lock.get("config_sha256") == config_hash == candidate.get("sha256")
    )
    decision = prerequisites.get("budget_decision", {})
    checks["budget_decision_identity"] = lock.get("budget_decision") == {
        "path": decision.get("decision_path"),
        "sha256": decision.get("decision_sha256"),
    }
    checks["sensitivity_identity"] = lock.get("sensitivity_evidence") == decision.get(
        "decision", {}
    ).get("sensitivity_evidence")
    hash_contract = prerequisites.get("candidate_contract", {})
    checks["candidate_contract_identity"] = lock.get("candidate_contract") == {
        "path": CANDIDATE_CONTRACT_PATH,
        "sha256": hash_contract.get("sha256"),
    }
    checks["authorization_policy_identity"] = lock.get("authorization_policy") == {
        "path": AUTHORIZATION_POLICY_PATH,
        "sha256": file_sha256(root / AUTHORIZATION_POLICY_PATH),
    }
    checks["runtime_profile_identity"] = lock.get("runtime_profile") == {
        "path": RUNTIME_PROFILE_PATH,
        "sha256": file_sha256(root / RUNTIME_PROFILE_PATH),
    }
    contract = prerequisites.get("contract", {})
    stored_contract = prerequisites.get("stored_contract_audit", {})
    checks["contract_identity_and_g1"] = (
        lock.get("contract_authorization", {}).get("audit")
        == {
            "path": stored_contract.get("path"),
            "sha256": stored_contract.get("sha256"),
        }
        and lock.get("contract_authorization", {}).get("artifact_identity")
        == contract.get("artifact_identity")
        and lock.get("contract_authorization", {}).get("status") == "pass"
        and lock.get("contract_authorization", {}).get("reviewed_count") == 12
        and lock.get("contract_authorization", {}).get("b_c_mismatch_count") == 0
    )
    candidate_value = prerequisites.get("final_candidate_config", {})
    checks["dataset_quarantine_manifest_hypotheses"] = (
        lock.get("dataset_sha256") == candidate_value.get("expected_dataset_sha256")
        and lock.get("quarantine_sha256")
        == {
            "questions": file_sha256(root / candidate_value["quarantine_path"]),
            "theories": file_sha256(root / candidate_value["theory_quarantine_path"]),
        }
        and lock.get("sample_manifest")
        == {
            "path": candidate_value.get("sample_manifest_path"),
            "sha256": file_sha256(root / candidate_value["sample_manifest_path"]),
        }
        and lock.get("primary_hypotheses")
        == {
            "path": PRIMARY_HYPOTHESES_PATH,
            "sha256": file_sha256(root / PRIMARY_HYPOTHESES_PATH),
        }
    )
    contract_value = _read_object(root / CANDIDATE_CONTRACT_PATH)
    checks["prompt_and_code_current"] = lock.get("prompt_sha256") == {
        Path(relative).name: digest
        for relative, digest in contract_value["prompt_sha256"].items()
    } and lock.get("inference_contract_sha256") == contract_value[
        "inference_contract_sha256"
    ]
    checks["final_paths_fresh"] = lock.get("final_paths") == prerequisites.get(
        "final_paths"
    ) and all(
        not (root / relative).exists()
        for relative in prerequisites.get("final_paths", {}).values()
    )
    blocking = [name for name, passed in checks.items() if not passed]
    if blocking:
        raise DatasetSchemaError(
            f"schema-5 Final semantic authorization failed: {blocking}"
        )
    return {
        "status": "pass",
        "checks": checks,
        "selected_max_tokens": selected,
        "model_calls_made": 0,
        "network_calls_made": 0,
    }


def _validated_candidate_spec(
    root: Path,
    candidates: Mapping[str, Any],
    budget: int,
    description: str,
) -> dict[str, Any]:
    spec = candidates.get(str(budget))
    if not isinstance(spec, Mapping):
        raise ValueError(f"{description} candidate for {budget} is missing")
    path = _resolve_relative_file(root, spec.get("path"), f"{description} candidate")
    digest = file_sha256(path)
    value = _read_object(path)
    client = value.get("client") if isinstance(value.get("client"), Mapping) else {}
    frozen_transport = (
        value.get("protocol_version") == PROTOCOL
        and value.get("attempt_id") == 2
        and value.get("execution_revision") == "attempt2-infrastructure-v1"
        and value.get("runtime_profile_path") == RUNTIME_PROFILE_PATH
        and value.get("runtime_profile_sha256")
        == file_sha256(root / RUNTIME_PROFILE_PATH)
        and client.get("provider") == "lmstudio"
        and client.get("model_id") == "qwen3.5-9b"
        and client.get("temperature") == 0
        and client.get("reasoning_effort") == "none"
        and client.get("timeout_seconds") == 360
        and client.get("max_retries") == 1
        and client.get("retry_backoff_seconds") == 1.0
        and client.get("fresh_cache_required") is True
        and client.get("use_cache") is True
        and all(
            isinstance(client.get(field), str) and client.get(field)
            for field in ("raw_dir", "cache_dir", "attempts_dir", "terminal_dir")
        )
    )
    if (
        digest != spec.get("sha256")
        or client.get("max_tokens") != budget
        or not frozen_transport
    ):
        raise ValueError(f"{description} candidate does not match budget/hash")
    return {"path": path.relative_to(root).as_posix(), "sha256": digest}


def _locked_file_matches(root: Path, relative: Any, expected_hash: Any) -> bool:
    try:
        path = _resolve_relative_file(root, relative, "locked file")
    except (DatasetSchemaError, TypeError, ValueError):
        return False
    return isinstance(expected_hash, str) and file_sha256(path) == expected_hash


def _resolve_relative_file(root: Path, relative: Any, description: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise DatasetSchemaError(f"{description} path must be non-empty and relative")
    candidate = (root / relative).resolve()
    if root.resolve() not in candidate.parents or not candidate.is_file():
        raise DatasetSchemaError(f"{description} is missing or escapes project")
    return candidate


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object at {path}:{line_number}")
        rows.append(value)
    return rows


def _finish_reason_from_row(root: Path, row: Mapping[str, Any]) -> str | None:
    raw_paths = row.get("raw_output_paths")
    if not isinstance(raw_paths, list) or len(raw_paths) != 1:
        return None
    path = Path(str(raw_paths[0]))
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not path.is_file():
        return None
    return _provider_finish_reason(_read_object(path))


def _provider_finish_reason(raw: Mapping[str, Any]) -> str | None:
    provider = raw.get("provider_payload")
    choices = provider.get("choices") if isinstance(provider, Mapping) else None
    if not isinstance(choices, list) or len(choices) != 1:
        return None
    choice = choices[0]
    value = choice.get("finish_reason") if isinstance(choice, Mapping) else None
    return value if isinstance(value, str) else None


def _contains_hidden_reasoning(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in {"reasoning", "reasoning_content"} and _has_content(item):
                return True
            if _contains_hidden_reasoning(item):
                return True
    elif isinstance(value, list):
        return any(_contains_hidden_reasoning(item) for item in value)
    return False


def contains_hidden_reasoning(value: Any) -> bool:
    """Public wrapper used by the current Contract auditor."""

    return _contains_hidden_reasoning(value)


def _has_content(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (Mapping, list, tuple)):
        return bool(value)
    return bool(value)


def _forbidden_fields_present(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in FORBIDDEN_DECISION_FIELDS:
                found.add(normalized)
            _collect_forbidden(item, found)
    elif isinstance(value, list):
        for item in value:
            _collect_forbidden(item, found)
    return found


def _collect_forbidden(value: Any, found: set[str]) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in FORBIDDEN_DECISION_FIELDS:
                found.add(normalized)
            _collect_forbidden(item, found)
    elif isinstance(value, list):
        for item in value:
            _collect_forbidden(item, found)


def _mapping_differences(
    left: Mapping[str, Any], right: Mapping[str, Any], prefix: str = ""
) -> set[str]:
    differences: set[str] = set()
    for key in set(left) | set(right):
        path = f"{prefix}.{key}" if prefix else str(key)
        left_value = left.get(key)
        right_value = right.get(key)
        if isinstance(left_value, Mapping) and isinstance(right_value, Mapping):
            differences.update(_mapping_differences(left_value, right_value, path))
        elif left_value != right_value:
            differences.add(path)
    return differences


def _is_strict_descendant(root: Path, base_commit: str) -> bool:
    head = _git_output(root, ["rev-parse", "HEAD"])
    if not head or head == base_commit:
        return False
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", base_commit, head],
        cwd=root,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def _tracked_worktree_clean(root: Path) -> bool:
    unstaged = subprocess.run(
        ["git", "diff", "--quiet"], cwd=root, capture_output=True, check=False
    )
    staged = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    return unstaged.returncode == 0 and staged.returncode == 0


def _git_output(root: Path, args: Iterable[str]) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else ""
