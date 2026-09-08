"""Offline physical-request reconstruction from immutable raw/cache artifacts."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def reconstruct_physical_accounting(
    records: Sequence[Mapping[str, Any]],
    *,
    cache_directories: Iterable[str | Path] | None = None,
) -> dict[str, Any]:
    """Separate method-attributed references from deduplicated physical calls.

    Result-row latency is intentionally never used as physical latency. The
    persisted non-cache-hit response is loaded from the cache artifact keyed by
    the exact request payload. Raw mtimes provide a backward-compatible wall
    clock estimate for artifacts written before explicit timestamps existed.
    """

    if not records:
        raise ValueError("physical accounting requires at least one result row")
    referenced_paths = [
        Path(str(path)).resolve()
        for row in records
        for path in (row.get("raw_output_paths") or [])
    ]
    if not referenced_paths:
        raise ValueError("result rows contain no raw artifact references")

    raw_by_key: dict[str, list[tuple[Path, Mapping[str, Any]]]] = defaultdict(list)
    referenced_path_to_key: dict[Path, str] = {}
    raw_directories: set[Path] = set()
    for path in sorted(set(referenced_paths)):
        raw = _read_object(path)
        payload = raw.get("request_payload")
        if not isinstance(payload, Mapping):
            raise ValueError(f"raw artifact has no request_payload object: {path}")
        key = request_identity(payload)
        raw_by_key[key].append((path, raw))
        referenced_path_to_key[path] = key
        raw_directories.add(path.parent)

    # Interrupted or repeated executions can leave more than one raw file for
    # one request identity. Include them and choose the earliest recoverable
    # real artifact rather than a later cache-zero method row.
    referenced_keys = set(raw_by_key)
    for directory in raw_directories:
        for path in directory.glob("*.json"):
            resolved = path.resolve()
            if resolved in referenced_path_to_key:
                continue
            try:
                raw = _read_object(resolved)
                payload = raw.get("request_payload")
                if not isinstance(payload, Mapping):
                    continue
                key = request_identity(payload)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if key in referenced_keys:
                raw_by_key[key].append((resolved, raw))

    cache_dirs = {Path(path).resolve() for path in (cache_directories or [])}
    cache_dirs.update(directory.parent / "cache" for directory in raw_directories)
    cache_by_key: dict[str, Mapping[str, Any]] = {}
    for key in sorted(referenced_keys):
        candidates = [directory / f"{key}.json" for directory in cache_dirs]
        existing = [path for path in candidates if path.exists()]
        if not existing:
            raise ValueError(f"cache artifact is missing for request identity {key}")
        values = [_read_object(path) for path in existing]
        first = values[0]
        if any(value != first for value in values[1:]):
            raise ValueError(f"conflicting cache artifacts for request identity {key}")
        cache_by_key[key] = first

    row_keys: list[list[str]] = []
    key_methods: dict[str, set[str]] = defaultdict(set)
    key_examples: dict[str, set[str]] = defaultdict(set)
    first_owner: dict[str, str] = {}
    for row in records:
        keys: list[str] = []
        for raw_path in row.get("raw_output_paths") or []:
            path = Path(str(raw_path)).resolve()
            key = referenced_path_to_key.get(path)
            if key is None:
                raw = _read_object(path)
                payload = raw.get("request_payload")
                if not isinstance(payload, Mapping):
                    raise ValueError(f"raw artifact has no request payload: {path}")
                key = request_identity(payload)
                referenced_path_to_key[path] = key
            keys.append(key)
            method = str(row.get("method"))
            key_methods[key].add(method)
            key_examples[key].add(str(row.get("example_id")))
            first_owner.setdefault(key, method)
        row_keys.append(keys)

    calls: dict[str, dict[str, Any]] = {}
    for key in sorted(referenced_keys):
        raw_candidates = raw_by_key[key]
        selected_path, selected_raw = min(
            raw_candidates, key=lambda item: _finish_epoch(item[0], item[1])
        )
        cache = cache_by_key[key]
        latency_ms = _physical_latency_ms(cache, selected_raw)
        finish_epoch = _finish_epoch(selected_path, selected_raw)
        start_epoch = _start_epoch(selected_raw, finish_epoch, latency_ms)
        prompt_tokens, completion_tokens, total_tokens = _physical_tokens(
            cache, selected_raw
        )
        calls[key] = {
            "cache_key": key,
            "stage": _request_stage(selected_raw),
            "raw_path": str(selected_path),
            "raw_candidate_count": len(raw_candidates),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "physical_latency_ms": latency_ms,
            "estimated_start_utc": _iso_utc(start_epoch),
            "estimated_finish_utc": _iso_utc(finish_epoch),
            "attributed_methods": sorted(key_methods[key]),
            "example_ids": sorted(key_examples[key]),
            "first_owner_method": first_owner.get(key),
        }

    method_attributed: dict[str, dict[str, Any]] = {}
    rows_by_method: dict[str, list[tuple[Mapping[str, Any], list[str]]]] = defaultdict(list)
    for row, keys in zip(records, row_keys):
        rows_by_method[str(row.get("method"))].append((row, keys))
    for method, pairs in sorted(rows_by_method.items()):
        rows = [row for row, _ in pairs]
        unique_keys = sorted({key for _, keys in pairs for key in keys})
        referenced_calls = [calls[key] for key in unique_keys]
        method_attributed[method] = {
            "method_result_row_count": len(rows),
            "logical_call_count": sum(int(row.get("call_count", 0)) for row in rows),
            "raw_reference_count": sum(len(keys) for _, keys in pairs),
            "attributed_prompt_tokens": sum(
                int((row.get("tokens") or {}).get("prompt", 0)) for row in rows
            ),
            "attributed_completion_tokens": sum(
                int((row.get("tokens") or {}).get("completion", 0)) for row in rows
            ),
            "attributed_total_tokens": sum(
                int((row.get("tokens") or {}).get("total", 0)) for row in rows
            ),
            "method_row_attributed_latency_ms_not_physical": sum(
                float(row.get("latency_ms", 0.0)) for row in rows
            ),
            "referenced_unique_physical_request_count_nonadditive": len(unique_keys),
            "referenced_physical_latency_ms_nonadditive": sum(
                float(call["physical_latency_ms"]) for call in referenced_calls
            ),
            "referenced_physical_total_tokens_nonadditive": sum(
                int(call["total_tokens"]) for call in referenced_calls
            ),
            "incremental_first_owner_physical_request_count": sum(
                call["first_owner_method"] == method for call in referenced_calls
            ),
        }

    stage_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for call in calls.values():
        stage_groups[str(call["stage"])].append(call)
    stage_stats = {
        stage: _physical_totals(group)
        for stage, group in sorted(stage_groups.items())
    }
    all_calls = list(calls.values())
    physical = _physical_totals(all_calls)
    starts = [_parse_epoch(str(call["estimated_start_utc"])) for call in all_calls]
    finishes = [_parse_epoch(str(call["estimated_finish_utc"])) for call in all_calls]
    physical.update(
        {
            "estimated_wall_clock_ms": (max(finishes) - min(starts)) * 1000.0,
            "estimated_start_utc": _iso_utc(min(starts)),
            "estimated_finish_utc": _iso_utc(max(finishes)),
            "stage_stats": stage_stats,
        }
    )
    attributed_tokens = {
        "prompt": sum(
            int((row.get("tokens") or {}).get("prompt", 0)) for row in records
        ),
        "completion": sum(
            int((row.get("tokens") or {}).get("completion", 0)) for row in records
        ),
        "total": sum(
            int((row.get("tokens") or {}).get("total", 0)) for row in records
        ),
    }
    raw_reference_count = sum(len(keys) for keys in row_keys)
    logical_call_count = sum(int(row.get("call_count", 0)) for row in records)
    return {
        "schema_version": 1,
        "accounting_contract": {
            "physical_identity": "sha256(canonical request_payload)",
            "physical_latency_source": "persisted non-cache-hit cache/raw artifact",
            "method_row_latency_is_physical": False,
            "nonadditive_method_physical_columns_are_explicitly_named": True,
        },
        "logical_method_attribution": {
            "method_result_row_count": len(records),
            "logical_call_count": logical_call_count,
            "raw_reference_count": raw_reference_count,
            "attributed_tokens": attributed_tokens,
            "methods": method_attributed,
        },
        "physical": physical,
        "reuse": {
            "physical_request_count": len(calls),
            "method_attributed_call_reference_count": logical_call_count,
            "raw_reference_count": raw_reference_count,
            "deduplicated_raw_reference_count": raw_reference_count - len(calls),
        },
        "physical_requests": [calls[key] for key in sorted(calls)],
    }


def request_identity(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def render_physical_accounting_markdown(report: Mapping[str, Any]) -> str:
    logical = report["logical_method_attribution"]
    physical = report["physical"]
    lines = [
        "# 离线物理调用与归因成本",
        "",
        "物理请求按唯一 request/cache identity 去重；方法归因列不可相加为物理成本。",
        "",
        "## 总计",
        "",
        "| 口径 | 请求/引用 | Prompt tokens | Completion tokens | Total tokens | Latency |",
        "|---|---:|---:|---:|---:|---:|",
        (
            "| Physical | "
            f"{physical['physical_request_count']} | {physical['prompt_tokens']} | "
            f"{physical['completion_tokens']} | {physical['total_tokens']} | "
            f"{physical['physical_latency_ms'] / 1000:.3f} s |"
        ),
        (
            "| Method-attributed | "
            f"{logical['logical_call_count']} | "
            f"{logical['attributed_tokens']['prompt']} | "
            f"{logical['attributed_tokens']['completion']} | "
            f"{logical['attributed_tokens']['total']} | N/A |"
        ),
        "",
        "## 方法归因（physical 引用列非加和）",
        "",
        "| Method | Logical calls | Attributed tokens | Referenced physical | Referenced physical latency | Incremental first-owner |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, item in logical["methods"].items():
        lines.append(
            f"| {method} | {item['logical_call_count']} | "
            f"{item['attributed_total_tokens']} | "
            f"{item['referenced_unique_physical_request_count_nonadditive']} | "
            f"{item['referenced_physical_latency_ms_nonadditive'] / 1000:.3f} s | "
            f"{item['incremental_first_owner_physical_request_count']} |"
        )
    lines.extend(
        [
            "",
            "方法行的 cache-sensitive latency 仅保存在 JSON 的归因命名字段中，"
            "不在本表冒充方法真实延迟。",
        ]
    )
    return "\n".join(lines) + "\n"


def _physical_totals(calls: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "physical_request_count": len(calls),
        "prompt_tokens": sum(int(call["prompt_tokens"]) for call in calls),
        "completion_tokens": sum(int(call["completion_tokens"]) for call in calls),
        "total_tokens": sum(int(call["total_tokens"]) for call in calls),
        "physical_latency_ms": sum(
            float(call["physical_latency_ms"]) for call in calls
        ),
    }


def _physical_latency_ms(
    cache: Mapping[str, Any], raw: Mapping[str, Any]
) -> float:
    for value in (raw.get("physical_latency_ms"), cache.get("latency_ms")):
        if isinstance(value, (int, float)) and value >= 0:
            return float(value)
    raise ValueError("physical latency is missing from both raw and cache artifacts")


def _physical_tokens(
    cache: Mapping[str, Any], raw: Mapping[str, Any]
) -> tuple[int, int, int]:
    provider = raw.get("provider_payload")
    usage = provider.get("usage") if isinstance(provider, Mapping) else None
    prompt = cache.get("prompt_tokens")
    completion = cache.get("completion_tokens")
    total = cache.get("total_tokens")
    if not all(isinstance(value, int) and value >= 0 for value in (prompt, completion, total)):
        if not isinstance(usage, Mapping):
            raise ValueError("physical token usage is missing")
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        total = usage.get("total_tokens")
    if not all(isinstance(value, int) and value >= 0 for value in (prompt, completion, total)):
        raise ValueError("physical token usage is invalid")
    if prompt + completion != total:
        raise ValueError("physical token total is inconsistent")
    return prompt, completion, total


def _request_stage(raw: Mapping[str, Any]) -> str:
    payload = raw.get("request_payload")
    messages = payload.get("messages") if isinstance(payload, Mapping) else None
    system = "\n".join(
        str(item.get("content", ""))
        for item in messages or []
        if isinstance(item, Mapping) and item.get("role") == "system"
    )
    if "REPAIR_REQUEST" in system:
        return "repair"
    for mode in ("cot_refine", "constrained", "direct", "plain", "cot"):
        if f"MODE={mode}" in system:
            return mode
    return "unknown"


def _finish_epoch(path: Path, raw: Mapping[str, Any]) -> float:
    value = raw.get("request_finished_utc")
    if isinstance(value, str):
        return _parse_epoch(value)
    return path.stat().st_mtime


def _start_epoch(raw: Mapping[str, Any], finish: float, latency_ms: float) -> float:
    value = raw.get("request_started_utc")
    if isinstance(value, str):
        return _parse_epoch(value)
    return finish - latency_ms / 1000.0


def _parse_epoch(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def _iso_utc(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value
