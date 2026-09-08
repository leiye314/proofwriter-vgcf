"""Frozen LM Studio runtime-profile binding for VGCF-2.3.1 Attempt-2."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .data import file_sha256
from .errors import DatasetSchemaError

PROTOCOL = "VGCF-2.3.1"
ATTEMPT_ID = 2
EXECUTION_REVISION = "attempt2-infrastructure-v1"
MODELS_ENDPOINT = "http://127.0.0.1:1234/api/v1/models"


@dataclass(frozen=True, slots=True)
class RuntimeProfile:
    schema_version: int
    protocol_version: str
    attempt_id: int
    execution_revision: str
    provider: str
    models_endpoint: str
    model_id: str
    context_length: int
    parallel: int
    timeout_seconds: float
    reasoning_effort: str


def load_runtime_profile(path: str | Path) -> RuntimeProfile:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DatasetSchemaError("runtime profile must be a JSON object")
    expected_fields = set(RuntimeProfile.__dataclass_fields__)
    if set(value) != expected_fields:
        raise DatasetSchemaError(
            "runtime profile fields differ from the frozen schema: "
            f"missing={sorted(expected_fields - set(value))} "
            f"unknown={sorted(set(value) - expected_fields)}"
        )
    for field_name in ("schema_version", "attempt_id", "context_length", "parallel"):
        item = value[field_name]
        if not isinstance(item, int) or isinstance(item, bool):
            raise DatasetSchemaError(f"runtime profile {field_name} must be an integer")
    timeout = value["timeout_seconds"]
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool):
        raise DatasetSchemaError("runtime profile timeout_seconds must be numeric")
    for field_name in (
        "protocol_version",
        "execution_revision",
        "provider",
        "models_endpoint",
        "model_id",
        "reasoning_effort",
    ):
        if not isinstance(value[field_name], str):
            raise DatasetSchemaError(f"runtime profile {field_name} must be a string")
    profile = RuntimeProfile(**{**value, "timeout_seconds": float(timeout)})
    if asdict(profile) != {
        "schema_version": 1,
        "protocol_version": PROTOCOL,
        "attempt_id": ATTEMPT_ID,
        "execution_revision": EXECUTION_REVISION,
        "provider": "lmstudio",
        "models_endpoint": MODELS_ENDPOINT,
        "model_id": "qwen3.5-9b",
        "context_length": 16384,
        "parallel": 1,
        "timeout_seconds": 360.0,
        "reasoning_effort": "none",
    }:
        raise DatasetSchemaError(_reload_message("runtime profile values changed"))
    return profile


def verify_runtime_profile_binding(root: Path, config: Any) -> RuntimeProfile:
    relative = getattr(config, "runtime_profile_path", None)
    expected_hash = getattr(config, "runtime_profile_sha256", None)
    if not isinstance(relative, str) or not relative:
        raise DatasetSchemaError("Attempt-2 config is missing runtime_profile_path")
    profile_path = (root / relative).resolve()
    try:
        profile_path.relative_to(root.resolve())
    except ValueError as exc:
        raise DatasetSchemaError("runtime profile must stay inside the project") from exc
    if not profile_path.is_file():
        raise DatasetSchemaError(f"runtime profile is missing: {relative}")
    actual_hash = file_sha256(profile_path)
    if expected_hash != actual_hash:
        raise DatasetSchemaError("runtime profile SHA-256 does not match the config binding")
    profile = load_runtime_profile(profile_path)
    client = getattr(config, "client", None)
    checks = {
        "protocol": getattr(config, "protocol_version", None)
        == profile.protocol_version,
        "attempt_id": getattr(config, "attempt_id", None) == profile.attempt_id,
        "execution_revision": getattr(config, "execution_revision", None)
        == profile.execution_revision,
        "provider": getattr(client, "provider", None) == profile.provider,
        "model": getattr(client, "model_id", None) == profile.model_id,
        "timeout": getattr(client, "timeout_seconds", None)
        == profile.timeout_seconds,
        "reasoning": getattr(client, "reasoning_effort", None)
        == profile.reasoning_effort,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise DatasetSchemaError(
            _reload_message(f"config/runtime-profile binding failed: {failed}")
        )
    return profile


def verify_lmstudio_runtime(
    root: Path,
    config: Any,
    *,
    opener: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """GET the local inventory and fail before any model client is constructed."""

    profile = verify_runtime_profile_binding(root.resolve(), config)
    open_url = opener or urllib.request.urlopen
    request = urllib.request.Request(profile.models_endpoint, method="GET")
    try:
        with open_url(request, timeout=10.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (
        OSError,
        TimeoutError,
        urllib.error.URLError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise DatasetSchemaError(
            _reload_message(f"LM Studio models endpoint is unavailable: {exc}")
        ) from exc
    report = verify_lmstudio_inventory(payload, profile)
    return {
        **report,
        "runtime_profile_path": getattr(config, "runtime_profile_path"),
        "runtime_profile_sha256": getattr(config, "runtime_profile_sha256"),
        "network_scope": "local_lmstudio_models_inventory_only",
    }


def verify_lmstudio_inventory(
    payload: Any, profile: RuntimeProfile
) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or "models" not in payload:
        raise DatasetSchemaError(_reload_message("models inventory schema is invalid"))
    models = payload.get("models")
    if not isinstance(models, list):
        raise DatasetSchemaError(_reload_message("models inventory is not a list"))
    matches: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for model in models:
        if not isinstance(model, Mapping):
            continue
        loaded = model.get("loaded_instances", [])
        if not isinstance(loaded, list):
            continue
        parent_ids = {
            model.get(field) for field in ("id", "key", "model_id")
        }
        for instance in loaded:
            if not isinstance(instance, Mapping):
                continue
            instance_ids = {
                instance.get(field) for field in ("id", "key", "model_id")
            }
            if profile.model_id in parent_ids | instance_ids:
                matches.append((model, instance))
    if len(matches) != 1:
        raise DatasetSchemaError(
            _reload_message(
                "expected exactly one loaded qwen3.5-9b instance, "
                f"found {len(matches)}"
            )
        )
    model, instance = matches[0]
    loaded_config = instance.get("config")
    if not isinstance(loaded_config, Mapping):
        raise DatasetSchemaError(_reload_message("loaded instance config is missing"))
    context_length = loaded_config.get("context_length")
    parallel = loaded_config.get("parallel")
    max_context_length = model.get("max_context_length")
    valid_integer_fields = all(
        isinstance(item, int) and not isinstance(item, bool)
        for item in (context_length, parallel, max_context_length)
    )
    if not valid_integer_fields:
        raise DatasetSchemaError(
            _reload_message("runtime context/parallel/max-context fields are invalid")
        )
    if (
        context_length != profile.context_length
        or parallel != profile.parallel
        or max_context_length < profile.context_length
    ):
        raise DatasetSchemaError(
            _reload_message(
                "loaded settings differ "
                f"(context_length={context_length}, parallel={parallel}, "
                f"max_context_length={max_context_length})"
            )
        )
    return {
        "status": "pass",
        "model_id": profile.model_id,
        "matching_loaded_instance_count": 1,
        "context_length": context_length,
        "parallel": parallel,
        "max_context_length": max_context_length,
        "automatic_reload_performed": False,
    }


def _reload_message(reason: str) -> str:
    return (
        f"{reason}. Reload qwen3.5-9b in LM Studio with "
        "context_length=16384 and parallel=1; automatic unload/reload is prohibited."
    )
