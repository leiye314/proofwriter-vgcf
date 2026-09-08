"""Evaluator-only VGCF-2.3 Contract-12 selection and audit.

This module may inspect gold labels for fixed-set selection and accuracy.  It is
never imported by prompt construction or ``MethodRunner``.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .audit_v2 import audit_v2_run
from .data import Example, file_sha256, load_examples
from .errors import DatasetSchemaError
from .experiment import load_config
from .metrics import aggregate_records
from .parsing import (
    parse_cot_refine_response,
    parse_cot_response,
    parse_label,
)
from .physical import reconstruct_physical_accounting
from .protocol import (
    KNOWN_DATASET_SHA256,
    load_sample_manifest,
    select_manifest_examples,
    verify_contract_lock_v2_3,
)

CONTRACT_SEED = 20260808
CONTRACT_SIZE = 12
CONTRACT_LABEL_QUOTA = 4
CONTRACT_METHODS = (
    "direct",
    "cot",
    "cot_refine",
    "constrained",
    "gate_cot",
    "repair_only",
    "vgcf2",
)
_LABELS = ("True", "False", "Unknown")
_PRIOR_SETS = ("probe_2", "dev_20", "confirmation_30", "dev_150")
_MANUAL_FIELDS = {
    "example_id",
    "reviewed",
    "initial_conclusion_a",
    "refined_conclusion_b",
    "final_label_c",
    "b_matches_c",
    "revised_a_to_b",
    "note",
}


def select_contract_examples(
    examples: Sequence[Example],
    *,
    excluded_example_ids: Iterable[str],
    excluded_theory_ids: Iterable[str],
    seed: int = CONTRACT_SEED,
) -> list[Example]:
    """Select four examples per label with deterministic hash tie-breaking."""

    excluded_examples = set(excluded_example_ids)
    excluded_theories = set(excluded_theory_ids)
    selected: list[Example] = []
    used_theories: set[str] = set()
    for label in _LABELS:
        pool = sorted(
            (
                example
                for example in examples
                if example.gold_label == label
                and example.example_id not in excluded_examples
                and example.theory_id not in excluded_theories
            ),
            key=lambda example: hashlib.sha256(
                f"{seed}:{example.example_id}".encode("utf-8")
            ).hexdigest(),
        )
        label_count = 0
        for example in pool:
            if example.theory_id in used_theories:
                continue
            selected.append(example)
            used_theories.add(example.theory_id)
            label_count += 1
            if label_count == CONTRACT_LABEL_QUOTA:
                break
        if label_count != CONTRACT_LABEL_QUOTA:
            raise DatasetSchemaError(
                f"Contract-12 selector cannot fill the {label} quota"
            )
    return selected


def contract_selection_fingerprint(examples: Sequence[Example], seed: int) -> str:
    payload = f"{seed}\n" + "\n".join(item.example_id for item in examples) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def audit_contract_manifest(
    contract_manifest: Mapping[str, Any],
    prior_manifest: Mapping[str, Any],
    examples: Sequence[Example],
) -> dict[str, Any]:
    """Check balance, exact reproducibility, and two-layer prior-set isolation."""

    contract_set = contract_manifest.get("sets", {}).get("contract_12")
    if not isinstance(contract_set, Mapping):
        raise DatasetSchemaError("Contract-12 manifest is missing sets.contract_12")
    selected = select_manifest_examples(examples, contract_manifest, "contract_12")
    expected_ids = [item.example_id for item in selected]
    by_id = {item.example_id: item for item in examples}
    prior_ids = {
        example_id
        for name in _PRIOR_SETS
        for example_id in prior_manifest["sets"][name]["example_ids"]
    }
    prior_theories = {by_id[item].theory_id for item in prior_ids}
    actual_theories = {item.theory_id for item in selected}
    label_counts = dict(sorted(Counter(item.gold_label for item in selected).items()))
    seed = int(contract_set.get("selection_seed", -1))
    reproduced = select_contract_examples(
        examples,
        excluded_example_ids=prior_ids,
        excluded_theory_ids=prior_theories,
        seed=seed,
    )
    fingerprint = contract_selection_fingerprint(reproduced, seed)
    checks = {
        "dataset_identity": contract_manifest.get("dataset_sha256")
        == KNOWN_DATASET_SHA256["meta-dev"],
        "size": len(selected) == CONTRACT_SIZE,
        "unique_examples": len(set(expected_ids)) == CONTRACT_SIZE,
        "unique_theories": len(actual_theories) == CONTRACT_SIZE,
        "balanced_labels": label_counts
        == {"False": 4, "True": 4, "Unknown": 4},
        "example_disjoint": not (set(expected_ids) & prior_ids),
        "theory_disjoint": not (actual_theories & prior_theories),
        "exact_selector_reproduction": [item.example_id for item in reproduced]
        == expected_ids,
        "selection_fingerprint": contract_set.get("selection_fingerprint")
        == fingerprint,
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "example_count": len(selected),
        "theory_count": len(actual_theories),
        "label_counts": label_counts,
        "depth_counts": dict(sorted(Counter(item.depth for item in selected).items())),
        "example_overlap_count": len(set(expected_ids) & prior_ids),
        "theory_overlap_count": len(actual_theories & prior_theories),
        "selection_seed": seed,
        "selection_fingerprint": fingerprint,
        "example_ids": expected_ids,
    }


def audit_historical_confirmation_v2_2(root: Path) -> dict[str, Any]:
    """Verify immutable VGCF-2.2 Confirmation evidence without rewriting it."""

    freeze = _read_object(root / "configs/v2_2_confirmation_artifact_freeze.json")
    artifact_root = root / str(freeze["root"])
    key_actual = {
        name: file_sha256(root / name) if (root / name).is_file() else None
        for name in freeze["key_artifact_sha256"]
    }
    tree_actual: dict[str, dict[str, Any]] = {}
    for relative, expected in freeze["trees"].items():
        directory = artifact_root / relative
        digest, count = _tree_hash(directory)
        tree_actual[relative] = {"file_count": count, "tree_sha256": digest}
    checks = {
        "key_artifact_sha256": key_actual == freeze["key_artifact_sha256"],
        "trees": tree_actual == freeze["trees"],
        "old_g1_marked_defect": freeze.get("historical_g1_status")
        == "audit_specification_defect",
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "actual_key_artifact_sha256": key_actual,
        "actual_trees": tree_actual,
        "historical_g1_status": freeze.get("historical_g1_status"),
    }


def automatic_object_conclusion_candidate(rationale: str) -> str | None:
    """Generate a review candidate only; never a final semantic judgment."""

    try:
        return parse_label(rationale)
    except Exception:
        return None


def evaluate_g1_abc_review(
    entries: Sequence[Mapping[str, Any]],
    expected_ids: Sequence[str],
    anchored_final_labels: Mapping[str, str],
) -> dict[str, Any]:
    """Evaluate human A/B/C annotations; gold is intentionally not an input."""

    errors: list[str] = []
    ids = [str(item.get("example_id")) for item in entries]
    if ids != list(expected_ids):
        errors.append("manual review IDs/order do not match Contract-12")
    reviewed_count = 0
    revision_count = 0
    mismatch_count = 0
    for index, entry in enumerate(entries):
        if set(entry) != _MANUAL_FIELDS:
            errors.append(f"entry {index} fields do not match the A/B/C schema")
            continue
        if entry.get("reviewed") is not True:
            if entry.get("reviewed") is not False:
                errors.append(f"entry {index} reviewed must be boolean")
            continue
        reviewed_count += 1
        a = entry.get("initial_conclusion_a")
        b = entry.get("refined_conclusion_b")
        c = entry.get("final_label_c")
        if a not in _LABELS or b not in _LABELS or c not in _LABELS:
            errors.append(f"entry {index} reviewed A/B/C values must be valid labels")
            continue
        example_id = str(entry["example_id"])
        if anchored_final_labels.get(example_id) != c:
            errors.append(f"entry {index} C does not equal the parsed anchored label")
        expected_consistency = b == c
        expected_revision = a != b
        if entry.get("b_matches_c") is not expected_consistency:
            errors.append(f"entry {index} b_matches_c is inconsistent with B and C")
        if entry.get("revised_a_to_b") is not expected_revision:
            errors.append(f"entry {index} revised_a_to_b is inconsistent with A and B")
        mismatch_count += int(not expected_consistency)
        revision_count += int(expected_revision)
    complete = reviewed_count == len(expected_ids)
    status = "invalid" if errors else ("complete" if complete else "pending")
    return {
        "status": status,
        "errors": errors,
        "reviewed_count": reviewed_count,
        "expected_count": len(expected_ids),
        "mismatch_count": mismatch_count,
        "revision_count": revision_count,
        "g1_consistent": complete and not errors and mismatch_count == 0,
        "definition": "G1 checks B == C; A != B is a legal revision",
        "gold_used_for_consistency": False,
    }


def evaluate_g1_gate(
    result_contract: Mapping[str, Any],
    manual: Mapping[str, Any],
    g1_spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Combine machine contract checks with final human A/B/C semantics."""

    checks = {
        "cot_contract_valid_12_of_12": result_contract["cot_valid_count"]
        == g1_spec["cot_contract_valid_required_count"],
        "cot_refine_contract_valid_12_of_12": result_contract[
            "cot_refine_valid_count"
        ]
        == g1_spec["cot_refine_contract_valid_required_count"],
        "all_rationales_manually_reviewed": manual["reviewed_count"]
        == g1_spec["manual_reviewed_required_count"],
        "no_last_label_fallback": result_contract["fallback_used_count"]
        <= g1_spec["last_label_fallback_max_count"],
    }
    machine_contract_pass = all(
        value
        for name, value in checks.items()
        if name != "all_rationales_manually_reviewed"
    )
    if not machine_contract_pass or manual["status"] == "invalid":
        status = "fail"
    elif manual["status"] != "complete":
        status = "pending"
    else:
        status = (
            "pass"
            if checks["all_rationales_manually_reviewed"]
            and manual["mismatch_count"] <= g1_spec["b_c_mismatch_max_count"]
            else "fail"
        )
    return {
        "status": status,
        "checks": checks,
        "manual_review": manual,
        "revision_count": manual["revision_count"],
        "b_c_mismatch_count": manual["mismatch_count"],
        "gold_used_for_consistency": False,
    }


def audit_contract12(
    project_root: str | Path,
    *,
    require_results: bool = False,
    preflight_only: bool = False,
    manual_review_path: str | Path | None = None,
) -> dict[str, Any]:
    """Audit the frozen VGCF-2.3 plan and, when available, its real results."""

    root = Path(project_root).resolve()
    paths = {
        "config": root / "configs/contract_12_v2_3.json",
        "manifest": root / "configs/v2_3_contract_samples.json",
        "prior_manifest": root / "configs/v2_2_dev_samples.json",
        "gate": root / "configs/contract_gate_v2_3.json",
        "manual_template": root / "configs/contract_manual_review_v2_3.json",
        "dataset": root
        / "data/proofwriter/proofwriter-dataset-V2020.12.3/OWA/depth-5/meta-dev.jsonl",
    }
    config, config_hash = load_config(paths["config"])
    examples = load_examples(paths["dataset"])
    contract_manifest = load_sample_manifest(paths["manifest"])
    prior_manifest = load_sample_manifest(paths["prior_manifest"])
    blocking: list[str] = []
    if config.protocol_version != "VGCF-2.3":
        blocking.append("Contract-12 config is not VGCF-2.3")
    if config.methods != CONTRACT_METHODS:
        blocking.append("Contract-12 does not contain the seven frozen methods")
    if config.sample_set != "contract_12" or config.sample_count != CONTRACT_SIZE:
        blocking.append("Contract-12 config sample contract is invalid")
    gate_spec = _read_object(paths["gate"])
    g1_spec = gate_spec.get("gates", {}).get("G1", {})
    protocol_spec = gate_spec.get("gates", {}).get("PROTOCOL", {})
    if (
        gate_spec.get("protocol") != "VGCF-2.3"
        or g1_spec.get("a_b_revision_is_failure") is not False
        or g1_spec.get("gold_used_for_consistency") is not False
        or g1_spec.get("automatic_extractor_is_final_judgment") is not False
    ):
        blocking.append("Contract-12 G1 specification is invalid")
    try:
        manifest_audit = audit_contract_manifest(
            contract_manifest, prior_manifest, examples
        )
    except Exception as exc:
        manifest_audit = {"status": "fail", "error": str(exc)}
    if manifest_audit["status"] != "pass":
        blocking.append("Contract-12 manifest audit failed")
    history = audit_historical_confirmation_v2_2(root)
    if history["status"] != "pass":
        blocking.append("historical VGCF-2.2 Confirmation hash audit failed")
    try:
        lock = verify_contract_lock_v2_3(
            config_hash=config_hash,
            lock_path=root / str(config.protocol_lock_path),
            project_root=root,
        )
        lock_audit = {"status": "pass", "protocol": lock["protocol"]}
    except Exception as exc:
        lock_audit = {"status": "fail", "error": str(exc)}
        blocking.append(f"Contract-12 protocol lock failed: {exc}")
    expected_ids = list(contract_manifest["sets"]["contract_12"]["example_ids"])
    template_value = _read_object(paths["manual_template"])
    template_audit = evaluate_g1_abc_review(
        template_value.get("entries", []), expected_ids, {}
    )
    if template_audit["status"] == "invalid":
        blocking.append("Contract-12 manual review template is invalid")
    preflight = {
        "status": "pass" if not blocking else "fail",
        "blocking": blocking,
        "config_sha256": config_hash,
        "manifest": manifest_audit,
        "historical_confirmation_v2_2": history,
        "protocol_lock": lock_audit,
        "manual_review_template": template_audit,
        "model_calls_made_by_auditor": 0,
    }
    result_path = root / config.output_path
    if preflight_only or not result_path.exists():
        if require_results and not result_path.exists():
            preflight["status"] = "fail"
            preflight["blocking"].append(f"Contract-12 results missing: {result_path}")
        return {
            "schema_version": 1,
            "protocol": "VGCF-2.3",
            "status": preflight["status"],
            "run_status": "not_run" if not result_path.exists() else "not_inspected",
            "preflight": preflight,
            "g1": {"status": "pending"},
            "manual_review_required_example_ids": expected_ids,
        }

    rows = _read_jsonl(result_path)
    result_contract = _audit_result_contract(rows, expected_ids, root)
    run_audit = audit_v2_run(result_path, paths["config"], root)
    reuse = _audit_shared_responses(rows, expected_ids, root)
    try:
        physical = reconstruct_physical_accounting(
            rows, cache_directories=[root / config.client.cache_dir]
        )
        physical_error = None
    except Exception as exc:
        physical = None
        physical_error = str(exc)
    manual_path = (
        Path(manual_review_path)
        if manual_review_path is not None
        else root / "outputs/v2_3_contract12/g1_manual_review.json"
    )
    if not manual_path.is_absolute():
        manual_path = (root / manual_path).resolve()
    manual_value = (
        _read_object(manual_path) if manual_path.exists() else template_value
    )
    manual = evaluate_g1_abc_review(
        manual_value.get("entries", []),
        expected_ids,
        result_contract["anchored_final_labels"],
    )
    g1 = evaluate_g1_gate(result_contract, manual, g1_spec)
    g1_status = g1["status"]
    infrastructure_error_count = sum(
        bool(row.get("infrastructure_error")) for row in rows
    )
    supporting_pass = (
        preflight["status"] == "pass"
        and result_contract["status"] == "pass"
        and run_audit["status"] == "pass"
        and reuse["status"] == "pass"
        and infrastructure_error_count
        <= protocol_spec["infrastructure_error_max_count"]
        and physical_error is None
    )
    if not supporting_pass or g1_status == "fail":
        status = "fail"
    elif g1_status == "pending":
        status = "pending"
    else:
        status = "pass"
    return {
        "schema_version": 1,
        "protocol": "VGCF-2.3",
        "status": status,
        "run_status": "audited",
        "preflight": preflight,
        "result_contract": result_contract,
        "run_audit": run_audit,
        "shared_response_audit": reuse,
        "physical_accounting": physical,
        "physical_accounting_error": physical_error,
        "infrastructure_error_count": infrastructure_error_count,
        "g1": g1,
        "automatic_review_candidates": result_contract["automatic_candidates"],
        "automatic_candidates_are_final_semantic_judgments": False,
        "manual_review_path": str(manual_path),
        "accuracy": aggregate_records(rows),
        "accuracy_separate_from_g1_consistency": True,
        "model_calls_made_by_auditor": 0,
    }


def _audit_result_contract(
    rows: Sequence[Mapping[str, Any]], expected_ids: Sequence[str], root: Path
) -> dict[str, Any]:
    expected_pairs = {(method, item) for item in expected_ids for method in CONTRACT_METHODS}
    actual_pairs = {(str(row.get("method")), str(row.get("example_id"))) for row in rows}
    indexed = {
        (str(row.get("method")), str(row.get("example_id"))): row for row in rows
    }
    cot_valid = 0
    refine_valid = 0
    fallback_count = 0
    final_labels: dict[str, str] = {}
    candidates: list[dict[str, Any]] = []
    parse_errors: list[dict[str, str]] = []
    for example_id in expected_ids:
        cot_row = indexed.get(("cot", example_id), {})
        refine_row = indexed.get(("cot_refine", example_id), {})
        cot_content = _last_content(cot_row, root)
        initial_content = _content_at(refine_row, root, 0)
        refined_content = _last_content(refine_row, root)
        try:
            cot = parse_cot_response(cot_content)
            cot_valid += 1
        except Exception as exc:
            cot = None
            parse_errors.append({"example_id": example_id, "stage": "cot", "error": str(exc)})
        try:
            initial = parse_cot_response(initial_content)
        except Exception as exc:
            initial = None
            parse_errors.append(
                {"example_id": example_id, "stage": "refine_initial", "error": str(exc)}
            )
        try:
            refined = parse_cot_refine_response(refined_content)
            refine_valid += 1
            final_labels[example_id] = refined.label
        except Exception as exc:
            refined = None
            parse_errors.append(
                {"example_id": example_id, "stage": "refined_review", "error": str(exc)}
            )
        for row, parsed in ((cot_row, cot), (refine_row, refined)):
            if parsed is None and row.get("predicted_label") in _LABELS:
                fallback_count += 1
        candidates.append(
            {
                "example_id": example_id,
                "automatic_initial_a_candidate": (
                    automatic_object_conclusion_candidate(initial.rationale)
                    if initial is not None
                    else None
                ),
                "automatic_refined_b_candidate": (
                    automatic_object_conclusion_candidate(refined.rationale)
                    if refined is not None
                    else None
                ),
                "anchored_final_c": refined.label if refined is not None else None,
                "manual_review_required": True,
            }
        )
    checks = {
        "row_count": len(rows) == CONTRACT_SIZE * len(CONTRACT_METHODS),
        "exact_method_example_pairs": actual_pairs == expected_pairs,
        "no_duplicate_pairs": len(indexed) == len(rows),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "cot_valid_count": cot_valid,
        "cot_refine_valid_count": refine_valid,
        "fallback_used_count": fallback_count,
        "anchored_final_labels": final_labels,
        "automatic_candidates": candidates,
        "automatic_candidates_are_final_semantic_judgments": False,
        "parse_errors": parse_errors,
    }


def _audit_shared_responses(
    rows: Sequence[Mapping[str, Any]], expected_ids: Sequence[str], root: Path
) -> dict[str, Any]:
    indexed = {
        (str(row.get("method")), str(row.get("example_id"))): row for row in rows
    }
    errors: list[str] = []
    for example_id in expected_ids:
        cot_paths = _resolved_raw_paths(indexed.get(("cot", example_id), {}), root)
        refine_paths = _resolved_raw_paths(
            indexed.get(("cot_refine", example_id), {}), root
        )
        if not cot_paths or len(refine_paths) < 2 or refine_paths[0] != cot_paths[0]:
            errors.append(f"{example_id}: CoT initial response is not shared with refine")
        formal_paths = {
            method: _resolved_raw_paths(indexed.get((method, example_id), {}), root)
            for method in ("constrained", "gate_cot", "repair_only", "vgcf2")
        }
        first_paths = {paths[0] for paths in formal_paths.values() if paths}
        if len(first_paths) != 1 or any(not paths for paths in formal_paths.values()):
            errors.append(f"{example_id}: constrained initial response is not shared")
        for method in ("gate_cot", "vgcf2"):
            row = indexed.get((method, example_id), {})
            paths = formal_paths[method]
            if row.get("fallback_used") is True and (
                not cot_paths or not paths or paths[-1] != cot_paths[0]
            ):
                errors.append(f"{example_id}: {method} fallback is not shared CoT")
    return {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "initial_cot_shared": not any("CoT initial" in item for item in errors),
        "constrained_initial_shared": not any("constrained initial" in item for item in errors),
        "fallback_cot_shared": not any("fallback" in item for item in errors),
    }


def _resolved_raw_paths(row: Mapping[str, Any], root: Path) -> list[Path]:
    result: list[Path] = []
    for value in row.get("raw_output_paths") or []:
        path = Path(str(value))
        result.append(path.resolve() if path.is_absolute() else (root / path).resolve())
    return result


def _content_at(row: Mapping[str, Any], root: Path, index: int) -> str:
    paths = _resolved_raw_paths(row, root)
    if not paths or index >= len(paths) or not paths[index].is_file():
        return ""
    return str(_read_object(paths[index]).get("content", ""))


def _last_content(row: Mapping[str, Any], root: Path) -> str:
    paths = _resolved_raw_paths(row, root)
    return _content_at(row, root, len(paths) - 1) if paths else ""


def _tree_hash(directory: Path) -> tuple[str, int]:
    files = (
        sorted(path for path in directory.rglob("*") if path.is_file())
        if directory.is_dir()
        else []
    )
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), len(files)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
