"""Reproducible experiment orchestration and structured result persistence."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .data import (
    balanced_sample,
    build_model_input,
    dataset_fingerprint,
    exclude_quarantined_examples,
    example_surface_features,
    file_sha256,
    fixed_sample,
    load_examples,
)
from .final_v2_3_1 import audit_final_manifest, enforce_final_theory_capacity
from .final_authorization_v2_3_1 import (
    verify_contract_candidate_runtime,
    verify_sensitivity_runtime,
)
from .gold import load_gold_programs, semantic_metrics
from .methods import METHODS, MethodRunner
from .model import ClientConfig, ModelRequestError, ModelResponse, create_client
from .protocol import (
    arm_final_test,
    audit_dataset_for_phase,
    audit_sample_disjointness,
    load_quarantine,
    load_sample_manifest,
    load_theory_quarantine,
    select_manifest_examples,
    verify_contract_lock_v2_3,
    verify_confirmation_lock,
)
from .runtime_profile_v2_3_1 import (
    EXECUTION_REVISION as ATTEMPT2_EXECUTION_REVISION,
    verify_lmstudio_runtime,
)
from .schema import Formalization


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    dataset_path: str
    output_path: str
    methods: tuple[str, ...]
    sample_count: int
    seed: int
    prompt_directory: str
    client: ClientConfig
    sampling_strategy: str = "random"
    phase: str = "software_validation"
    max_questions_per_theory: int = 1
    quarantine_path: str | None = None
    theory_quarantine_path: str | None = None
    sample_manifest_path: str | None = None
    expected_sample_manifest_sha256: str | None = None
    sample_set: str | None = None
    protocol_lock_path: str | None = None
    expected_dataset_sha256: str | None = None
    gold_evaluation: bool = False
    protocol_version: str = "VGCF-2.1"
    attempt_id: int | None = None
    execution_revision: str | None = None
    runtime_profile_path: str | None = None
    runtime_profile_sha256: str | None = None


def summarize_transport_evidence(
    events: tuple[ModelResponse | ModelRequestError, ...],
    *,
    output_contract_failed: bool,
) -> dict[str, Any]:
    """Flatten per-logical-call transport evidence into one result-row binding."""

    logical_request_ids: list[str] = []
    attempt_paths: list[str] = []
    terminal_paths: list[str] = []
    physical_elapsed_ms = 0.0
    error_category: str | None = None
    provider_usage_available = bool(events)
    for event in events:
        if isinstance(event, ModelRequestError):
            logical_request_ids.append(event.logical_request_id)
            attempt_paths.extend(event.transport_attempt_paths)
            terminal_paths.append(event.terminal_evidence_path)
            physical_elapsed_ms += event.physical_elapsed_ms
            error_category = event.error_category
            provider_usage_available = False
            continue
        logical_request_ids.extend(event.logical_request_ids)
        attempt_paths.extend(event.transport_attempt_paths)
        terminal_paths.extend(event.terminal_evidence_paths)
        physical_elapsed_ms += event.physical_elapsed_ms
        if event.error_category is not None:
            error_category = event.error_category
        provider_usage_available = (
            provider_usage_available and event.provider_usage_available
        )
    if output_contract_failed:
        error_category = "output_contract_failure"
    return {
        "logical_request_ids": logical_request_ids,
        "transport_attempt_count": len(attempt_paths),
        "transport_attempt_paths": attempt_paths,
        "terminal_evidence_path": terminal_paths[-1] if terminal_paths else None,
        "terminal_evidence_paths": terminal_paths,
        "physical_elapsed_ms": physical_elapsed_ms,
        "error_category": error_category,
        "provider_usage_available": provider_usage_available,
    }


def load_config(path: str | Path) -> tuple[ExperimentConfig, str]:
    config_path = Path(path)
    raw = config_path.read_bytes()
    value = json.loads(raw)
    config = normalize_config(value)
    identity = audit_dataset_for_phase(
        config.phase, config.dataset_path, config_only=True
    )
    if (
        config.expected_dataset_sha256 is not None
        and config.expected_dataset_sha256 != identity.sha256
    ):
        raise ValueError("expected_dataset_sha256 does not match the dataset bytes")
    return config, hashlib.sha256(raw).hexdigest()


def normalize_config(value: Mapping[str, Any]) -> ExperimentConfig:
    """Strictly apply the loader defaults and return one semantic config object."""

    if not isinstance(value, dict):
        raise ValueError("experiment config must be a JSON object")
    if value.get("artifact_status") == "frozen_historical":
        raise ValueError(
            "frozen historical config is audit-only and cannot be passed to the v2 runner"
        )
    required = {
        "dataset_path",
        "output_path",
        "methods",
        "sample_count",
        "seed",
        "prompt_directory",
        "client",
    }
    optional = {
        "sampling_strategy",
        "phase",
        "max_questions_per_theory",
        "quarantine_path",
        "theory_quarantine_path",
        "sample_manifest_path",
        "expected_sample_manifest_sha256",
        "sample_set",
        "protocol_lock_path",
        "expected_dataset_sha256",
        "gold_evaluation",
        "protocol_version",
        "attempt_id",
        "execution_revision",
        "runtime_profile_path",
        "runtime_profile_sha256",
    }
    unknown = set(value) - required - optional
    missing = required - set(value)
    if missing or unknown:
        raise ValueError(f"config missing={sorted(missing)} unknown={sorted(unknown)}")
    _require_json_type(value, "dataset_path", str)
    _require_json_type(value, "output_path", str)
    _require_json_type(value, "sample_count", int)
    _require_json_type(value, "seed", int)
    _require_json_type(value, "prompt_directory", str)
    if not isinstance(value["methods"], list) or any(
        not isinstance(method, str) for method in value["methods"]
    ):
        raise ValueError("config.methods must be an array of strings")
    methods = tuple(value["methods"])
    if not methods or any(method not in METHODS for method in methods):
        raise ValueError(f"methods must be selected from {METHODS}")
    client_value = value["client"]
    if not isinstance(client_value, dict):
        raise ValueError("config.client must be an object")
    valid_client_fields = {field.name for field in fields(ClientConfig)}
    client_unknown = set(client_value) - valid_client_fields
    if client_unknown:
        raise ValueError(f"unknown client config fields: {sorted(client_unknown)}")
    client_missing = {"provider", "model_id"} - set(client_value)
    if client_missing:
        raise ValueError(f"client config missing fields: {sorted(client_missing)}")
    client_defaults = asdict(ClientConfig(provider="", model_id=""))
    normalized_client = {**client_defaults, **client_value}
    for field_name in (
        "provider",
        "model_id",
        "base_url",
        "api_key_env",
        "cache_dir",
        "raw_dir",
    ):
        _require_json_type(normalized_client, field_name, str, prefix="config.client")
    for field_name in ("attempts_dir", "terminal_dir", "reasoning_effort"):
        _require_nullable_string(
            normalized_client, field_name, prefix="config.client"
        )
    for field_name in ("max_retries", "max_tokens"):
        _require_json_type(
            normalized_client, field_name, int, prefix="config.client"
        )
    for field_name in (
        "timeout_seconds",
        "temperature",
        "input_cost_per_million",
        "output_cost_per_million",
        "retry_backoff_seconds",
    ):
        _require_json_number(normalized_client, field_name, prefix="config.client")
        normalized_client[field_name] = float(normalized_client[field_name])
    for field_name in ("use_cache", "fresh_cache_required"):
        _require_json_type(
            normalized_client, field_name, bool, prefix="config.client"
        )
    for field_name in (
        "sampling_strategy",
        "phase",
        "protocol_version",
    ):
        if field_name in value:
            _require_json_type(value, field_name, str)
    for field_name in (
        "quarantine_path",
        "theory_quarantine_path",
        "sample_manifest_path",
        "expected_sample_manifest_sha256",
        "sample_set",
        "protocol_lock_path",
        "expected_dataset_sha256",
        "execution_revision",
        "runtime_profile_path",
        "runtime_profile_sha256",
    ):
        _require_nullable_string(value, field_name)
    for field_name in ("max_questions_per_theory", "attempt_id"):
        if field_name in value and value[field_name] is not None:
            _require_json_type(value, field_name, int)
    if "gold_evaluation" in value:
        _require_json_type(value, "gold_evaluation", bool)
    config = ExperimentConfig(
        dataset_path=value["dataset_path"],
        output_path=value["output_path"],
        methods=methods,
        sample_count=value["sample_count"],
        seed=value["seed"],
        prompt_directory=value["prompt_directory"],
        client=ClientConfig(**normalized_client),
        sampling_strategy=value.get("sampling_strategy", "random"),
        phase=value.get("phase", "software_validation"),
        max_questions_per_theory=value.get("max_questions_per_theory", 1),
        quarantine_path=value.get("quarantine_path"),
        theory_quarantine_path=value.get("theory_quarantine_path"),
        sample_manifest_path=value.get("sample_manifest_path"),
        expected_sample_manifest_sha256=value.get(
            "expected_sample_manifest_sha256"
        ),
        sample_set=value.get("sample_set"),
        protocol_lock_path=value.get("protocol_lock_path"),
        expected_dataset_sha256=value.get("expected_dataset_sha256"),
        gold_evaluation=value.get("gold_evaluation", False),
        protocol_version=value.get("protocol_version", "VGCF-2.1"),
        attempt_id=value.get("attempt_id"),
        execution_revision=value.get("execution_revision"),
        runtime_profile_path=value.get("runtime_profile_path"),
        runtime_profile_sha256=value.get("runtime_profile_sha256"),
    )
    if config.sampling_strategy not in {"random", "balanced", "manifest"}:
        raise ValueError("sampling_strategy must be random, balanced, or manifest")
    if config.max_questions_per_theory < 1:
        raise ValueError("max_questions_per_theory must be at least 1")
    if config.sampling_strategy == "manifest" and not (
        config.sample_manifest_path and config.sample_set
    ):
        raise ValueError("manifest sampling requires sample_manifest_path and sample_set")
    if (
        config.protocol_version == "VGCF-2.3.1"
        and config.phase == "final_test"
        and config.expected_sample_manifest_sha256 is None
    ):
        raise ValueError("VGCF-2.3.1 Final requires a frozen sample manifest hash")
    if config.phase == "final_test" and not all(
        (
            config.quarantine_path,
            config.theory_quarantine_path,
            config.protocol_lock_path,
            config.expected_dataset_sha256,
        )
    ):
        raise ValueError(
            "final_test requires both quarantines, expected dataset hash, and protocol lock"
        )
    return config


def canonical_normalized_config(
    value: ExperimentConfig | Mapping[str, Any],
) -> dict[str, Any]:
    """Return the JSON-compatible canonical semantic representation."""

    config = value if isinstance(value, ExperimentConfig) else normalize_config(value)
    normalized = asdict(config)
    normalized["methods"] = list(config.methods)
    return normalized


def config_semantic_sha256(
    value: ExperimentConfig | Mapping[str, Any],
) -> str:
    normalized = canonical_normalized_config(value)
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_json_type(
    value: Mapping[str, Any],
    field_name: str,
    expected: type,
    *,
    prefix: str = "config",
) -> None:
    item = value.get(field_name)
    if not isinstance(item, expected) or (
        expected is int and isinstance(item, bool)
    ):
        raise ValueError(f"{prefix}.{field_name} must be {expected.__name__}")


def _require_json_number(
    value: Mapping[str, Any], field_name: str, *, prefix: str
) -> None:
    item = value.get(field_name)
    if not isinstance(item, (int, float)) or isinstance(item, bool):
        raise ValueError(f"{prefix}.{field_name} must be a number")


def _require_nullable_string(
    value: Mapping[str, Any], field_name: str, *, prefix: str = "config"
) -> None:
    if field_name not in value:
        return
    item = value[field_name]
    if item is not None and not isinstance(item, str):
        raise ValueError(f"{prefix}.{field_name} must be string or null")


def run_experiment(
    config: ExperimentConfig,
    config_hash: str,
    project_root: str | Path,
) -> list[dict[str, Any]]:
    root = Path(project_root).resolve()
    dataset_path = _resolve(root, config.dataset_path)
    output_path = _resolve(root, config.output_path)
    prompt_directory = _resolve(root, config.prompt_directory)
    if (
        config.protocol_version == "VGCF-2.3.1"
        and config.sample_set == "cot4096_length_5"
    ):
        verify_sensitivity_runtime(root, config_hash=config_hash, config=config)
    if (
        config.protocol_version == "VGCF-2.3.1"
        and config.sample_set == "contract_12"
    ):
        verify_contract_candidate_runtime(
            root, config_hash=config_hash, config=config
        )
    if (
        config.protocol_version == "VGCF-2.2"
        and config.sample_set == "confirmation_30"
    ):
        verify_confirmation_lock(
            config_hash=config_hash,
            lock_path=(
                _resolve(root, config.protocol_lock_path)
                if config.protocol_lock_path is not None
                else None
            ),
            project_root=root,
        )
    if (
        config.protocol_version == "VGCF-2.3"
        and config.sample_set == "contract_12"
    ):
        verify_contract_lock_v2_3(
            config_hash=config_hash,
            lock_path=(
                _resolve(root, config.protocol_lock_path)
                if config.protocol_lock_path is not None
                else None
            ),
            project_root=root,
        )
    final_test_armed = False
    identity = audit_dataset_for_phase(
        config.phase,
        dataset_path,
        config_only=config.phase == "final_test",
    )
    if (
        config.expected_dataset_sha256
        and config.expected_dataset_sha256 != identity.sha256
    ):
        raise ValueError("dataset identity changed after config parsing")
    examples = load_examples(dataset_path)
    loaded_examples = list(examples)
    loaded_example_count = len(examples)
    quarantine: dict[str, tuple[str, ...]] = {}
    theory_quarantine_ids: frozenset[str] = frozenset()
    quarantine_excluded_count = 0
    quarantine_question_only_excluded_count = 0
    if config.phase == "final_test":
        assert config.quarantine_path is not None
        assert config.theory_quarantine_path is not None
        quarantine = load_quarantine(_resolve(root, config.quarantine_path))
        theory_quarantine = load_theory_quarantine(
            _resolve(root, config.theory_quarantine_path)
        )
        theory_quarantine_ids = theory_quarantine.theory_ids
        question_only = exclude_quarantined_examples(examples, quarantine, ())
        quarantine_question_only_excluded_count = len(examples) - len(question_only)
        eligible = exclude_quarantined_examples(
            examples, quarantine, theory_quarantine_ids
        )
        quarantine_excluded_count = len(examples) - len(eligible)
        examples = eligible
        enforce_final_theory_capacity(
            examples,
            sample_count=config.sample_count,
            max_questions_per_theory=config.max_questions_per_theory,
        )
    sample_audit: Mapping[str, Any] | None = None
    if config.sampling_strategy == "manifest":
        assert config.sample_manifest_path is not None
        assert config.sample_set is not None
        sample_manifest_path = _resolve(root, config.sample_manifest_path)
        if (
            config.expected_sample_manifest_sha256 is not None
            and file_sha256(sample_manifest_path)
            != config.expected_sample_manifest_sha256
        ):
            raise ValueError("sample manifest hash changed after config freezing")
        sample_manifest = load_sample_manifest(sample_manifest_path)
        sample_audit = audit_sample_disjointness(sample_manifest, examples)
        if config.protocol_version == "VGCF-2.3.1" and config.phase == "final_test":
            assert config.quarantine_path is not None
            assert config.theory_quarantine_path is not None
            final_manifest_audit = audit_final_manifest(
                sample_manifest,
                loaded_examples,
                dataset_sha256=identity.sha256,
                question_quarantine=quarantine,
                theory_quarantine_ids=theory_quarantine_ids,
                question_quarantine_path=_resolve(root, config.quarantine_path),
                theory_quarantine_path=_resolve(
                    root, config.theory_quarantine_path
                ),
            )
            if final_manifest_audit["status"] != "pass":
                raise ValueError(
                    "Final-300 manifest exact-reproduction audit failed: "
                    f"{final_manifest_audit['checks']}"
                )
            sample_audit = {
                **sample_audit,
                "final_manifest_checks": final_manifest_audit["checks"],
            }
        selected = select_manifest_examples(
            examples, sample_manifest, config.sample_set
        )
        if len(selected) != config.sample_count:
            raise ValueError(
                f"sample set {config.sample_set!r} has {len(selected)} examples, "
                f"config requested {config.sample_count}"
            )
    elif config.sampling_strategy == "balanced":
        selected = balanced_sample(
            examples,
            config.sample_count,
            config.seed,
            config.max_questions_per_theory,
        )
    else:
        selected = fixed_sample(
            examples,
            config.sample_count,
            config.seed,
            config.max_questions_per_theory,
        )
    if config.phase == "final_test":
        arm_final_test(
            config_hash=config_hash,
            protocol_version=config.protocol_version,
            dataset_path=dataset_path,
            prompt_directory=prompt_directory,
            quarantine_path=(
                _resolve(root, config.quarantine_path)
                if config.quarantine_path is not None
                else None
            ),
            theory_quarantine_path=(
                _resolve(root, config.theory_quarantine_path)
                if config.theory_quarantine_path is not None
                else None
            ),
            lock_path=(
                _resolve(root, config.protocol_lock_path)
                if config.protocol_lock_path is not None
                else None
            ),
            sample_manifest_path=(
                _resolve(root, config.sample_manifest_path)
                if config.sample_manifest_path is not None
                else None
            ),
            expected_sample_manifest_sha256=(
                config.expected_sample_manifest_sha256
            ),
        )
        final_test_armed = True
        identity = audit_dataset_for_phase(
            config.phase, dataset_path, final_test_armed=True
        )
    gold_programs = (
        load_gold_programs(dataset_path, [example.example_id for example in selected])
        if config.gold_evaluation
        else {}
    )
    client_dict = asdict(config.client)
    for directory_field in ("cache_dir", "raw_dir"):
        client_dict[directory_field] = str(
            _resolve(root, client_dict[directory_field])
        )
    for directory_field in ("attempts_dir", "terminal_dir"):
        if client_dict[directory_field] is not None:
            client_dict[directory_field] = str(
                _resolve(root, client_dict[directory_field])
            )
    if config.execution_revision == ATTEMPT2_EXECUTION_REVISION:
        verify_lmstudio_runtime(root, config)
    client = create_client(ClientConfig(**client_dict))
    runner = MethodRunner(
        client,
        prompt_directory,
        phase=config.phase,
        final_test_armed=final_test_armed,
    )
    git_commit = current_git_commit(root)
    timestamp = datetime.now(timezone.utc).isoformat()
    semantic_config_hash = config_semantic_sha256(config)
    records: list[dict[str, Any]] = []
    cache_hit_results = 0
    for example_index, example in enumerate(selected, start=1):
        model_input = build_model_input(example)
        model_input_hash = _model_input_hash(model_input)
        for method in config.methods:
            transport_cursor = client.transport_evidence_cursor()
            output = runner.run(method, example.model_view())
            transport_evidence = summarize_transport_evidence(
                client.transport_evidence_since(transport_cursor),
                output_contract_failed=output.method_error,
            )
            cache_hit_results += int(output.cached)
            semantic = None
            if output.parsed_structure is not None and example.example_id in gold_programs:
                semantic = semantic_metrics(
                    Formalization.from_dict(output.parsed_structure),
                    gold_programs[example.example_id].formalization,
                )
            final_label_correct = (
                output.predicted_label == example.gold_label
                if output.predicted_label in {"True", "False", "Unknown"}
                else False
            )
            records.append(
                {
                    "example_id": example.example_id,
                    "depth": example.depth,
                    "gold_label": example.gold_label,
                    "predicted_label": output.predicted_label,
                    "method": method,
                    "raw_model_response": output.raw_model_response,
                    "parsed_structure": output.parsed_structure,
                    "validation_errors": list(output.validation_errors),
                    "repair_triggered": output.repair_triggered,
                    "repair_attempted": output.repair_attempted,
                    "repair_improved": output.repair_improved,
                    "repair_regression": output.repair_regression,
                    "repair_regression_layers": list(output.repair_regression_layers),
                    "pre_repair_result": output.pre_repair_result,
                    "post_repair_result": output.post_repair_result,
                    "proof_trace": list(output.proof_trace),
                    "json_parse_valid": output.json_parse_valid,
                    "strict_schema_valid": output.strict_schema_valid,
                    "normalized_schema_valid": output.normalized_schema_valid,
                    "schema_valid": output.schema_valid,
                    "hard_validator_pass": output.hard_validator_pass,
                    "soft_issue_count": output.soft_issue_count,
                    "solver_executable": output.solver_executable,
                    "strict_solver_executable": output.strict_solver_executable,
                    "normalized_solver_executable": (
                        output.normalized_solver_executable
                    ),
                    "singleton_if_normalized_count": (
                        output.singleton_if_normalized_count
                    ),
                    "shadow_solver_label": output.shadow_solver_label,
                    "answer_from_solver": output.answer_from_solver,
                    "gold_semantic_match": (
                        semantic["exact_match"] if semantic is not None else None
                    ),
                    "gold_semantic_normalized_diagnostic_match": (
                        semantic["normalized_diagnostic_match"]
                        if semantic is not None
                        else None
                    ),
                    "gold_semantic_renaming_invariant_match": (
                        semantic.get("renaming_invariant_semantic_match")
                        if semantic is not None
                        else None
                    ),
                    "formalization_semantics": semantic,
                    "final_label_correct": final_label_correct,
                    "route_source": output.route_source,
                    "coverage": output.coverage,
                    "fallback_used": output.fallback_used,
                    "infrastructure_error": output.infrastructure_error,
                    "method_error": output.method_error,
                    "cot_refine_contract_valid": (
                        output.cot_refine_contract_valid
                    ),
                    "call_count": output.call_count,
                    "latency_ms": output.latency_ms,
                    "latency_semantics": (
                        "method_attributed_cache_sensitive_not_physical"
                    ),
                    "tokens": {
                        "prompt": output.prompt_tokens,
                        "completion": output.completion_tokens,
                        "total": output.total_tokens,
                    },
                    "cost_usd": output.cost_usd,
                    "model_id": output.model_id,
                    "config_hash": config_hash,
                    "config_file_sha256": config_hash,
                    "config_semantic_sha256": semantic_config_hash,
                    "prompt_hash": output.prompt_hash,
                    "git_commit": git_commit,
                    "cached": output.cached,
                    "error": output.error,
                    "source_profile": example.source_profile,
                    "model_input": model_input,
                    "model_input_hash": model_input_hash,
                    "raw_output_paths": list(output.raw_output_paths),
                    "initial_response_hash": output.initial_response_hash,
                    "initial_raw_output_path": output.initial_raw_output_path,
                    "initial_cache_key": output.initial_cache_key,
                    "answer_response_hash": output.answer_response_hash,
                    "answer_raw_output_path": output.answer_raw_output_path,
                    "answer_cache_key": output.answer_cache_key,
                    **transport_evidence,
                    "phase": config.phase,
                    "source_split": example.source_split,
                    "theory_id": example.theory_id,
                    "dataset_sha256": identity.sha256,
                    "sample_set": config.sample_set,
                    "run_timestamp_utc": timestamp,
                }
            )
            _write_jsonl(output_path, records)
        print(
            json.dumps(
                {
                    "sample_index": example_index,
                    "sample_total": len(selected),
                    "example_id": example.example_id,
                    "completed_method_results": len(records),
                    "cache_hit_method_results": cache_hit_results,
                    "remaining_method_results": (
                        len(selected) * len(config.methods) - len(records)
                    ),
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
            flush=True,
        )
    _write_jsonl(output_path, records)
    manifest_path = output_path.with_suffix(".manifest.json")
    normalized_config = canonical_normalized_config(config)
    manifest_path.write_text(
        json.dumps(
            {
                "config": normalized_config,
                "normalized_config": normalized_config,
                "config_hash": config_hash,
                "config_file_sha256": config_hash,
                "config_semantic_sha256": semantic_config_hash,
                "dataset_fingerprint": dataset_fingerprint(selected),
                "dataset_path": str(dataset_path),
                "dataset_sha256": file_sha256(dataset_path),
                "dataset_split": identity.split,
                "loaded_example_count": loaded_example_count,
                "eligible_example_count": len(examples),
                "selected_example_count": len(selected),
                "record_count": len(records),
                "git_commit": git_commit,
                "model_input_whitelist": ["theory", "question"],
                "sampling_strategy": config.sampling_strategy,
                "phase": config.phase,
                "max_questions_per_theory": config.max_questions_per_theory,
                "unique_theory_count": len(
                    {example.theory_id or example.example_id for example in selected}
                ),
                "quarantine_id_count": len(quarantine),
                "quarantine_theory_count": len(theory_quarantine_ids),
                "quarantine_question_only_excluded_count": quarantine_question_only_excluded_count,
                "quarantine_excluded_count": quarantine_excluded_count,
                "quarantine_path": config.quarantine_path,
                "theory_quarantine_path": config.theory_quarantine_path,
                "sample_manifest_path": config.sample_manifest_path,
                "sample_set": config.sample_set,
                "sample_disjointness_audit": sample_audit,
                "gold_evaluation": config.gold_evaluation,
                "selected_examples": [
                    {
                        "example_id": example.example_id,
                        "theory_id": example.theory_id,
                        "source_split": example.source_split,
                        "gold_label": example.gold_label,
                        "depth": example.depth,
                        "model_input_hash": _model_input_hash(build_model_input(example)),
                        **example_surface_features(example),
                    }
                    for example in selected
                ],
                "timestamp_utc": timestamp,
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return records


def current_git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "UNCOMMITTED"


def _redacted_config(config: ExperimentConfig) -> dict[str, Any]:
    value = canonical_normalized_config(config)
    value["client"]["api_key_env"] = config.client.api_key_env
    return value


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _write_jsonl(path: Path, records: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _model_input_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
