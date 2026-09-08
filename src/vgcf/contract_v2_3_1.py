"""Current-code/current-budget Contract-12 audit for VGCF-2.3.1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .audit_v2 import audit_v2_run
from .contract_v2_3 import (
    CONTRACT_METHODS,
    _audit_result_contract,
    _audit_shared_responses,
    _read_jsonl,
    audit_contract_manifest,
    evaluate_g1_abc_review,
    evaluate_g1_gate,
)
from .data import file_sha256, load_examples
from .experiment import (
    canonical_normalized_config,
    config_semantic_sha256,
    current_git_commit,
    load_config,
    normalize_config,
)
from .final_authorization_v2_3_1 import (
    CONTRACT_CANDIDATE_PATHS,
    PROTOCOL,
    audit_budget_decision,
    audit_contract_candidate_binding,
    audit_contract_candidate_freeze,
    canonical_json_bytes,
    contains_hidden_reasoning,
    tree_snapshot,
)
from .metrics import aggregate_records
from .physical import reconstruct_physical_accounting
from .protocol import load_sample_manifest

GATE_PATH = "configs/contract_gate_v2_3_1.json"
MANUAL_TEMPLATE_PATH = "configs/contract_manual_review_v2_3_1.json"
CONTRACT_MANIFEST_PATH = "configs/v2_3_contract_samples.json"
PRIOR_MANIFEST_PATH = "configs/v2_2_dev_samples.json"
DATASET_PATH = (
    "data/proofwriter/proofwriter-dataset-V2020.12.3/OWA/depth-5/"
    "meta-dev.jsonl"
)


def selected_contract_paths(
    root: Path, selected_budget: int
) -> dict[str, Path]:
    """Resolve all selected Contract artifact paths from its frozen config."""

    config_path = root / CONTRACT_CANDIDATE_PATHS[selected_budget]
    config, _ = load_config(config_path)
    output = root / config.output_path
    output_root = output.parent
    return {
        "config": config_path,
        "results": output,
        "run_manifest": output.with_suffix(".manifest.json"),
        "raw": root / config.client.raw_dir,
        "cache": root / config.client.cache_dir,
        "attempts": root / str(config.client.attempts_dir),
        "terminal": root / str(config.client.terminal_dir),
        "manual_review": output_root / "g1_manual_review.json",
        "audit": output_root / "contract_audit.json",
    }


def audit_current_contract(
    project_root: str | Path,
    *,
    require_results: bool = False,
    preflight_only: bool = False,
    manual_review_path: str | Path | None = None,
) -> dict[str, Any]:
    """Recompute the selected Contract/G1 evidence without trusting claims."""

    root = Path(project_root).resolve()
    freeze = audit_contract_candidate_freeze(root)
    decision = audit_budget_decision(root)
    if decision.get("status") != "pass":
        return {
            "schema_version": 2,
            "protocol": PROTOCOL,
            "status": "fail" if decision.get("status") == "fail" else "pending",
            "run_status": "not_run",
            "blocking": list(decision.get("blocking", [])),
            "candidate_freeze": freeze,
            "budget_decision": {
                "status": decision.get("status"),
                "blocking": decision.get("blocking", []),
            },
            "g1": {"status": "pending"},
            "model_calls_made_by_auditor": 0,
            "network_calls_made_by_auditor": 0,
        }
    selected_budget = int(decision["selected_max_tokens"])
    paths = selected_contract_paths(root, selected_budget)
    config, config_hash = load_config(paths["config"])
    expected_candidate = decision["decision"]["selected_contract_candidate"]
    binding = audit_contract_candidate_binding(root, config_hash, selected_budget)
    contract_manifest = load_sample_manifest(root / CONTRACT_MANIFEST_PATH)
    prior_manifest = load_sample_manifest(root / PRIOR_MANIFEST_PATH)
    examples = load_examples(root / DATASET_PATH)
    manifest_audit = audit_contract_manifest(
        contract_manifest, prior_manifest, examples
    )
    expected_ids = list(contract_manifest["sets"]["contract_12"]["example_ids"])
    gate = _read_object(root / GATE_PATH)
    g1_spec = gate.get("gates", {}).get("G1", {})
    protocol_spec = gate.get("gates", {}).get("PROTOCOL", {})
    template = _read_object(root / MANUAL_TEMPLATE_PATH)
    template_audit = evaluate_g1_abc_review(
        template.get("entries", []), expected_ids, {}
    )
    checks = {
        "candidate_freeze": freeze.get("status") == "pass",
        "selected_candidate_hash": expected_candidate.get("path")
        == paths["config"].relative_to(root).as_posix()
        and expected_candidate.get("sha256") == config_hash,
        "selected_budget": config.client.max_tokens == selected_budget,
        "current_code_lock": binding.get("status") == "pass",
        "protocol_and_methods": config.protocol_version == PROTOCOL
        and config.methods == CONTRACT_METHODS,
        "same_contract_12": config.sample_set == "contract_12"
        and config.sample_count == 12
        and config.sample_manifest_path == CONTRACT_MANIFEST_PATH,
        "manifest_reproduction": manifest_audit.get("status") == "pass",
        "gate_schema": gate.get("protocol") == PROTOCOL
        and g1_spec.get("manual_reviewed_required_count") == 12
        and g1_spec.get("b_c_mismatch_max_count") == 0
        and g1_spec.get("a_b_revision_is_failure") is False
        and g1_spec.get("gold_used_for_consistency") is False,
        "manual_template": template.get("protocol") == PROTOCOL
        and template_audit.get("status") == "pending",
    }
    if preflight_only:
        checks["fresh_result_raw_cache_audit_review"] = not any(
            paths[name].exists()
            for name in (
                "results",
                "run_manifest",
                "raw",
                "cache",
                "attempts",
                "terminal",
                "audit",
                "manual_review",
            )
        )
    blocking = [name for name, passed in checks.items() if not passed]
    preflight = {
        "status": "pass" if not blocking else "fail",
        "checks": checks,
        "blocking": blocking,
        "selected_max_tokens": selected_budget,
        "selected_config_path": paths["config"].relative_to(root).as_posix(),
        "selected_config_sha256": config_hash,
        "candidate_binding": binding,
        "manifest": manifest_audit,
        "manual_review_template": template_audit,
    }
    if preflight_only or not paths["results"].is_file():
        missing_blocking = list(blocking)
        if require_results and not paths["results"].is_file():
            missing_blocking.append("selected Contract results are missing")
        return {
            "schema_version": 2,
            "protocol": PROTOCOL,
            "status": (
                "fail"
                if missing_blocking
                else ("pass" if preflight_only else "pending")
            ),
            "run_status": (
                "not_run" if not paths["results"].exists() else "not_inspected"
            ),
            "blocking": missing_blocking,
            "preflight": preflight,
            "budget_decision": {
                "status": "pass",
                "path": decision["decision_path"],
                "sha256": decision["decision_sha256"],
                "selected_max_tokens": selected_budget,
            },
            "g1": {"status": "pending"},
            "manual_review_required_example_ids": expected_ids,
            "model_calls_made_by_auditor": 0,
            "network_calls_made_by_auditor": 0,
        }

    rows = _read_jsonl(paths["results"])
    result_contract = _audit_result_contract(rows, expected_ids, root)
    run_audit = audit_v2_run(paths["results"], paths["config"], root)
    shared = _audit_shared_responses(rows, expected_ids, root)
    try:
        physical = reconstruct_physical_accounting(
            rows, cache_directories=[paths["cache"]]
        )
        physical_error = None
    except Exception as exc:
        physical = None
        physical_error = str(exc)
    manual_path = (
        Path(manual_review_path)
        if manual_review_path is not None
        else paths["manual_review"]
    )
    if not manual_path.is_absolute():
        manual_path = (root / manual_path).resolve()
    manual_value = _read_object(manual_path) if manual_path.is_file() else template
    manual = evaluate_g1_abc_review(
        manual_value.get("entries", []),
        expected_ids,
        result_contract["anchored_final_labels"],
    )
    g1 = evaluate_g1_gate(result_contract, manual, g1_spec)
    infrastructure_error_count = sum(
        row.get("infrastructure_error") is True for row in rows
    )
    hidden_reasoning_files: list[str] = []
    if paths["raw"].is_dir():
        for raw_path in sorted(
            item for item in paths["raw"].rglob("*") if item.is_file()
        ):
            try:
                raw_value = _read_object(raw_path)
            except Exception:
                hidden_reasoning_files.append(
                    raw_path.relative_to(root).as_posix() + ":unreadable"
                )
                continue
            if contains_hidden_reasoning(raw_value.get("provider_payload")):
                hidden_reasoning_files.append(raw_path.relative_to(root).as_posix())
    raw_tree = tree_snapshot(paths["raw"])
    cache_tree = tree_snapshot(paths["cache"])
    attempt_tree = tree_snapshot(paths["attempts"])
    terminal_tree = tree_snapshot(paths["terminal"])
    transport_evidence = _audit_transport_evidence(
        root, rows, paths, selected_budget
    )
    run_manifest = (
        _read_object(paths["run_manifest"])
        if paths["run_manifest"].is_file()
        else {}
    )
    head = current_git_commit(root)
    row_commits = {str(row.get("git_commit")) for row in rows}
    row_config_hashes = {str(row.get("config_hash")) for row in rows}
    row_file_hashes = {
        str(row.get("config_file_sha256")) for row in rows
    }
    row_semantic_hashes = {
        str(row.get("config_semantic_sha256")) for row in rows
    }
    semantic_hash = config_semantic_sha256(config)
    normalized_config = canonical_normalized_config(config)
    try:
        manifest_normalized = canonical_normalized_config(
            normalize_config(run_manifest.get("config", {}))
        )
        manifest_config_error = None
    except (TypeError, ValueError) as exc:
        manifest_normalized = None
        manifest_config_error = str(exc)
    run_manifest_checks = {
        "exists": paths["run_manifest"].is_file(),
        "config_file_sha256": run_manifest.get("config_hash") == config_hash
        and run_manifest.get("config_file_sha256") == config_hash
        and row_config_hashes == {config_hash}
        and row_file_hashes == {config_hash},
        "config_semantic_sha256": run_manifest.get("config_semantic_sha256")
        == semantic_hash
        and row_semantic_hashes == {semantic_hash},
        "config_shared_normalizer": manifest_config_error is None
        and manifest_normalized == normalized_config
        and run_manifest.get("normalized_config") == normalized_config,
        "record_count_84": run_manifest.get("record_count") == 84,
        "selected_example_count_12": run_manifest.get("selected_example_count") == 12,
        "git_commit_current": run_manifest.get("git_commit") == head
        and row_commits == {head},
    }
    supporting_checks = {
        "preflight": preflight["status"] == "pass",
        "result_contract": result_contract["status"] == "pass"
        and result_contract["cot_valid_count"] == 12
        and result_contract["cot_refine_valid_count"] == 12
        and result_contract["fallback_used_count"] == 0,
        "run_audit": run_audit["status"] == "pass",
        "shared_response": shared["status"] == "pass",
        "infrastructure_error_zero": infrastructure_error_count
        <= protocol_spec.get("infrastructure_error_max_count", -1),
        "hidden_reasoning_zero": len(hidden_reasoning_files)
        <= protocol_spec.get("hidden_reasoning_max_count", -1),
        "physical_accounting": physical_error is None,
        "append_only_transport_evidence": transport_evidence["status"] == "pass",
        "run_manifest": all(run_manifest_checks.values()),
        "raw_tree_nonempty": raw_tree["exists"] and raw_tree["file_count"] > 0,
        "cache_tree_nonempty": cache_tree["exists"] and cache_tree["file_count"] > 0,
    }
    supporting_blocking = [
        name for name, passed in supporting_checks.items() if not passed
    ]
    if supporting_blocking or g1["status"] == "fail":
        status = "fail"
    elif g1["status"] == "pending":
        status = "pending"
    else:
        status = "pass"
    artifact_identity = {
        "config": {
            "path": paths["config"].relative_to(root).as_posix(),
            "sha256": config_hash,
        },
        "max_tokens": selected_budget,
        "results": {
            "path": paths["results"].relative_to(root).as_posix(),
            "sha256": file_sha256(paths["results"]),
        },
        "run_manifest": {
            "path": paths["run_manifest"].relative_to(root).as_posix(),
            "sha256": (
                file_sha256(paths["run_manifest"])
                if paths["run_manifest"].is_file()
                else None
            ),
        },
        "raw_tree": raw_tree,
        "cache_tree": cache_tree,
        "attempt_tree": attempt_tree,
        "terminal_tree": terminal_tree,
        "g1_manual_review": {
            "path": manual_path.relative_to(root).as_posix(),
            "sha256": file_sha256(manual_path) if manual_path.is_file() else None,
        },
        "inference_contract_sha256": binding["inference_contract_sha256"],
        "git_commit": head,
    }
    return {
        "schema_version": 2,
        "protocol": PROTOCOL,
        "status": status,
        "run_status": "audited",
        "blocking": supporting_blocking,
        "preflight": preflight,
        "budget_decision": {
            "status": "pass",
            "path": decision["decision_path"],
            "sha256": decision["decision_sha256"],
            "selected_max_tokens": selected_budget,
        },
        "artifact_identity": artifact_identity,
        "result_contract": result_contract,
        "run_audit": run_audit,
        "run_manifest_checks": run_manifest_checks,
        "manifest_config_error": manifest_config_error,
        "shared_response_audit": shared,
        "physical_accounting": physical,
        "physical_accounting_error": physical_error,
        "transport_evidence": transport_evidence,
        "infrastructure_error_count": infrastructure_error_count,
        "hidden_reasoning_count": len(hidden_reasoning_files),
        "hidden_reasoning_files": hidden_reasoning_files,
        "g1": g1,
        "automatic_review_candidates": result_contract["automatic_candidates"],
        "automatic_candidates_are_final_semantic_judgments": False,
        "accuracy": aggregate_records(rows),
        "accuracy_separate_from_g1_consistency": True,
        "model_calls_made_by_auditor": 0,
        "network_calls_made_by_auditor": 0,
    }


def contract_audit_bytes(report: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(report)


def _audit_transport_evidence(
    root: Path,
    rows: list[dict[str, Any]],
    paths: Mapping[str, Path],
    selected_budget: int,
) -> dict[str, Any]:
    """Bind successful Contract calls to terminal and attempt hash chains."""

    actual = {
        name: {
            item.relative_to(root).as_posix()
            for item in paths[name].rglob("*")
            if item.is_file()
        }
        if paths[name].is_dir()
        else set()
        for name in ("raw", "cache", "attempts", "terminal")
    }
    referenced_terminal = {
        _relative_evidence_path(root, item)
        for row in rows
        for item in row.get("terminal_evidence_paths", [])
        if isinstance(item, str)
    }
    referenced_attempts = {
        _relative_evidence_path(root, item)
        for row in rows
        for item in row.get("transport_attempt_paths", [])
        if isinstance(item, str)
    }
    terminal_attempts: set[str | None] = set()
    terminal_raw: set[str | None] = set()
    terminal_cache: set[str | None] = set()
    blocking: list[str] = []
    terminal_count = 0
    attempt_count = 0
    for relative in sorted(actual["terminal"]):
        try:
            terminal = _read_object(root / relative)
        except Exception as exc:
            blocking.append(f"unreadable terminal {relative}: {exc}")
            continue
        terminal_count += 1
        logical_id = terminal.get("logical_request_id")
        expected_terminal = None
        if isinstance(logical_id, str):
            try:
                expected_terminal = (
                    (paths["terminal"] / f"{logical_id}.json")
                    .resolve()
                    .relative_to(root)
                    .as_posix()
                )
            except ValueError:
                expected_terminal = None
        if (
            terminal.get("schema_version") != 1
            or terminal.get("terminal_outcome") != "success"
            or terminal.get("terminal_error_category") is not None
            or not isinstance(logical_id, str)
            or relative != expected_terminal
        ):
            blocking.append(f"invalid success terminal: {relative}")
        specs = terminal.get("attempt_artifacts")
        if not isinstance(specs, list) or not 1 <= len(specs) <= 2:
            blocking.append(f"invalid attempt list: {relative}")
            specs = []
        if terminal.get("transport_attempt_count") != len(specs):
            blocking.append(f"terminal attempt count mismatch: {relative}")
        attempt_count += len(specs)
        for index, spec in enumerate(specs, 1):
            if not isinstance(spec, Mapping):
                blocking.append(f"invalid attempt binding: {relative}")
                continue
            attempt_relative = _relative_evidence_path(root, spec.get("path"))
            terminal_attempts.add(attempt_relative)
            attempt_path = root / str(attempt_relative)
            if (
                attempt_relative not in actual["attempts"]
                or not attempt_path.is_file()
                or spec.get("sha256") != file_sha256(attempt_path)
            ):
                blocking.append(f"attempt path/hash mismatch: {relative}#{index}")
                continue
            try:
                attempt = _read_object(attempt_path)
            except Exception as exc:
                blocking.append(
                    f"unreadable attempt {attempt_relative}: {exc}"
                )
                continue
            if (
                attempt.get("schema_version") != 1
                or attempt.get("logical_request_id") != logical_id
                or attempt.get("attempt_index") != index
                or attempt.get("max_attempts") != 2
                or attempt.get("provider") != "lmstudio"
                or attempt.get("model") != "qwen3.5-9b"
                or attempt.get("timeout_seconds") != 360
                or attempt.get("max_tokens") != selected_budget
            ):
                blocking.append(
                    f"attempt runtime identity mismatch: {attempt_relative}"
                )
        raw_relative = _relative_evidence_path(
            root, terminal.get("success_raw_path")
        )
        cache_relative = _relative_evidence_path(
            root, terminal.get("success_cache_path")
        )
        terminal_raw.add(raw_relative)
        terminal_cache.add(cache_relative)
        for label, relative_path, expected_hash in (
            ("raw", raw_relative, terminal.get("success_raw_sha256")),
            ("cache", cache_relative, terminal.get("success_cache_sha256")),
        ):
            file_path = root / str(relative_path)
            if (
                relative_path not in actual[label]
                or not file_path.is_file()
                or expected_hash != file_sha256(file_path)
            ):
                blocking.append(f"terminal success {label} mismatch: {relative}")
    checks = {
        "terminal_set_exact": referenced_terminal == actual["terminal"],
        "attempt_set_exact": referenced_attempts
        == terminal_attempts
        == actual["attempts"],
        "raw_set_exact": terminal_raw == actual["raw"],
        "cache_set_exact": terminal_cache == actual["cache"],
        "attempt_count_exact": attempt_count == len(actual["attempts"]),
        "nonempty": terminal_count > 0 and attempt_count > 0,
        "all_terminals_valid": not blocking,
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "blocking": blocking,
        "terminal_count": terminal_count,
        "attempt_count": attempt_count,
    }


def _relative_evidence_path(root: Path, value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    candidate = Path(value)
    candidate = (
        candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    )
    try:
        return candidate.relative_to(root).as_posix()
    except ValueError:
        return None


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value
