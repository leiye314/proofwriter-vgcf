"""Unified OpenAI-compatible and deterministic Mock model clients."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .controlled_language import formalize_controlled
from .errors import ModelError
from .ir_v2 import formalization_to_v2_dict
from .solver import ForwardChainingSolver


@dataclass(frozen=True, slots=True)
class ClientConfig:
    provider: str
    model_id: str
    base_url: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    timeout_seconds: float = 60.0
    max_retries: int = 2
    temperature: float = 0.0
    max_tokens: int = 2000
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0
    cache_dir: str = "outputs/cache"
    raw_dir: str = "outputs/raw"
    use_cache: bool = True
    reasoning_effort: str | None = None
    attempts_dir: str | None = None
    terminal_dir: str | None = None
    retry_backoff_seconds: float = 1.0
    fresh_cache_required: bool = False


@dataclass(frozen=True, slots=True)
class ModelResponse:
    content: str
    model_id: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    cached: bool
    raw_output_path: str | None
    provider_payload: Mapping[str, Any]
    raw_output_paths: tuple[str, ...] = ()
    cache_key: str | None = None
    logical_request_ids: tuple[str, ...] = ()
    transport_attempt_count: int = 0
    transport_attempt_paths: tuple[str, ...] = ()
    terminal_evidence_path: str | None = None
    terminal_evidence_paths: tuple[str, ...] = ()
    physical_elapsed_ms: float = 0.0
    error_category: str | None = None
    provider_usage_available: bool = False

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["provider_payload"] = dict(self.provider_payload)
        value["raw_output_paths"] = list(self.raw_output_paths)
        value["logical_request_ids"] = list(self.logical_request_ids)
        value["transport_attempt_paths"] = list(self.transport_attempt_paths)
        value["terminal_evidence_paths"] = list(self.terminal_evidence_paths)
        return value


@dataclass(frozen=True, slots=True)
class _TransportTrace:
    logical_request_id: str
    request_tag: str
    cache_key: str
    attempt_paths: tuple[str, ...]
    physical_elapsed_ms: float
    terminal_path: str


@dataclass(frozen=True, slots=True)
class _RequestResult:
    content: str
    usage: Mapping[str, Any]
    provider_payload: Mapping[str, Any]
    trace: _TransportTrace | None = None


class ModelRequestError(ModelError):
    """A failed logical request with complete append-only transport evidence."""

    def __init__(
        self,
        message: str,
        *,
        logical_request_id: str,
        transport_attempt_paths: Sequence[str],
        terminal_evidence_path: str,
        physical_elapsed_ms: float,
        error_category: str,
    ) -> None:
        super().__init__(message)
        self.logical_request_id = logical_request_id
        self.transport_attempt_paths = tuple(transport_attempt_paths)
        self.terminal_evidence_path = terminal_evidence_path
        self.physical_elapsed_ms = physical_elapsed_ms
        self.error_category = error_category


class _ProviderSchemaError(ModelError):
    pass


class _EmptyVisibleContentError(ModelError):
    pass


class _HiddenReasoningError(ModelError):
    pass


class ResponseCache:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(payload: Mapping[str, Any]) -> str:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def get(self, key: str) -> ModelResponse | None:
        path = self.directory / f"{key}.json"
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if "raw_output_paths" not in value:
                legacy_path = value.get("raw_output_path")
                value["raw_output_paths"] = [legacy_path] if legacy_path else []
            value["raw_output_paths"] = tuple(value["raw_output_paths"])
            if "cache_key" not in value:
                value["cache_key"] = key
            for tuple_field in (
                "logical_request_ids",
                "transport_attempt_paths",
                "terminal_evidence_paths",
            ):
                value[tuple_field] = tuple(value.get(tuple_field, ()))
            value.setdefault("transport_attempt_count", 0)
            value.setdefault("terminal_evidence_path", None)
            value.setdefault("physical_elapsed_ms", 0.0)
            value.setdefault("error_category", None)
            value.setdefault("provider_usage_available", False)
            response = ModelResponse(**value)
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            raise ModelError(f"corrupt response cache entry {path}: {exc}") from exc
        return replace(response, cached=True, latency_ms=0.0, cache_key=key)

    def put(self, key: str, response: ModelResponse) -> Path:
        path = self.directory / f"{key}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(response.to_dict(), ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(path)
        return path.resolve()


class BaseModelClient:
    def __init__(self, config: ClientConfig) -> None:
        self.config = config
        self._transport_events: list[ModelResponse | ModelRequestError] = []
        if (config.attempts_dir is None) != (config.terminal_dir is None):
            raise ModelError(
                "attempts_dir and terminal_dir must either both be set or both be absent"
            )
        if config.attempts_dir is not None and config.max_retries > 1:
            raise ModelError(
                "append-only transport evidence permits at most two physical attempts"
            )
        cache_directory = Path(config.cache_dir)
        if (
            config.fresh_cache_required
            and cache_directory.exists()
            and any(cache_directory.iterdir())
        ):
            raise ModelError(
                f"fresh cache is required but cache directory is not empty: {cache_directory}"
            )
        self.cache = ResponseCache(config.cache_dir)
        self.raw_directory = Path(config.raw_dir)
        self.raw_directory.mkdir(parents=True, exist_ok=True)
        self.attempts_directory = (
            Path(config.attempts_dir) if config.attempts_dir is not None else None
        )
        self.terminal_directory = (
            Path(config.terminal_dir) if config.terminal_dir is not None else None
        )
        if self.attempts_directory is not None:
            self.attempts_directory.mkdir(parents=True, exist_ok=True)
        if self.terminal_directory is not None:
            self.terminal_directory.mkdir(parents=True, exist_ok=True)

    def transport_evidence_cursor(self) -> int:
        """Return an opaque cursor for logical-call evidence emitted after it."""

        return len(self._transport_events)

    def transport_evidence_since(
        self, cursor: int
    ) -> tuple[ModelResponse | ModelRequestError, ...]:
        if not isinstance(cursor, int) or cursor < 0 or cursor > len(
            self._transport_events
        ):
            raise ValueError("invalid transport evidence cursor")
        return tuple(self._transport_events[cursor:])

    def _record_transport_event(
        self, event: ModelResponse | ModelRequestError
    ) -> None:
        self._transport_events.append(event)

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        request_tag: str,
        response_format: Mapping[str, Any] | None = None,
    ) -> ModelResponse:
        request_payload: dict[str, Any] = {
            "provider": self.config.provider,
            "model": self.config.model_id,
            "messages": [dict(message) for message in messages],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.reasoning_effort is not None:
            request_payload["reasoning_effort"] = self.config.reasoning_effort
        if response_format is not None:
            request_payload["response_format"] = dict(response_format)
        key = self.cache.key(request_payload)
        if self.config.use_cache:
            cached = self.cache.get(key)
            if cached is not None and not (
                self.config.reasoning_effort == "none" and _has_hidden_reasoning(cached)
            ):
                self._record_transport_event(cached)
                return cached
            if self.config.reasoning_effort == "none":
                legacy_payload = dict(request_payload)
                legacy_payload.pop("reasoning_effort")
                legacy_cached = self.cache.get(self.cache.key(legacy_payload))
                if legacy_cached is not None and not _has_hidden_reasoning(legacy_cached):
                    self._record_transport_event(legacy_cached)
                    return legacy_cached
        started_utc = datetime.now(timezone.utc)
        started = time.perf_counter()
        request_parameters = inspect.signature(self._request).parameters
        try:
            if "request_tag" in request_parameters and "cache_key" in request_parameters:
                raw_request_result = self._request(
                    request_payload,
                    request_tag=request_tag,
                    cache_key=key,
                )
            else:
                raw_request_result = self._request(request_payload)  # type: ignore[call-arg]
        except ModelRequestError as exc:
            self._record_transport_event(exc)
            raise
        request_result = _coerce_request_result(raw_request_result)
        content = request_result.content
        usage = request_result.usage
        provider_payload = request_result.provider_payload
        if (
            self.config.reasoning_effort == "none"
            and _payload_has_hidden_reasoning(provider_payload)
        ):
            raise ModelError(
                "provider returned hidden reasoning while reasoning_effort='none'"
            )
        latency_ms = (time.perf_counter() - started) * 1000.0
        finished_utc = datetime.now(timezone.utc)
        prompt_tokens = int(usage.get("prompt_tokens", _estimate_tokens(messages)))
        completion_tokens = int(usage.get("completion_tokens", max(1, len(content) // 4)))
        total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens))
        cost = (
            prompt_tokens * self.config.input_cost_per_million
            + completion_tokens * self.config.output_cost_per_million
        ) / 1_000_000
        raw_path = self._save_raw(
            request_tag,
            key,
            request_payload,
            content,
            provider_payload,
            started_utc=started_utc,
            finished_utc=finished_utc,
            physical_latency_ms=latency_ms,
        )
        actual_model_id = provider_payload.get("model", self.config.model_id)
        if not isinstance(actual_model_id, str) or not actual_model_id:
            actual_model_id = self.config.model_id
        trace = request_result.trace
        terminal_path = trace.terminal_path if trace is not None else None
        response = ModelResponse(
            content=content,
            model_id=actual_model_id,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost,
            cached=False,
            raw_output_path=str(raw_path),
            provider_payload=provider_payload,
            raw_output_paths=(str(raw_path),),
            cache_key=key,
            logical_request_ids=(trace.logical_request_id,) if trace else (),
            transport_attempt_count=len(trace.attempt_paths) if trace else 0,
            transport_attempt_paths=trace.attempt_paths if trace else (),
            terminal_evidence_path=terminal_path,
            terminal_evidence_paths=(terminal_path,) if terminal_path else (),
            physical_elapsed_ms=(trace.physical_elapsed_ms if trace else latency_ms),
            error_category=None,
            provider_usage_available=_usage_available(usage),
        )
        cache_path: Path | None = None
        if self.config.use_cache:
            cache_path = self.cache.put(key, response)
        if trace is not None:
            self._write_success_terminal(
                trace,
                raw_path=raw_path,
                cache_path=cache_path,
                provider_usage_available=response.provider_usage_available,
            )
        self._record_transport_event(response)
        return response

    def _request(
        self,
        payload: Mapping[str, Any],
        *,
        request_tag: str | None = None,
        cache_key: str | None = None,
    ) -> _RequestResult:
        raise NotImplementedError

    def _save_raw(
        self,
        request_tag: str,
        key: str,
        request_payload: Mapping[str, Any],
        content: str,
        provider_payload: Mapping[str, Any],
        *,
        started_utc: datetime,
        finished_utc: datetime,
        physical_latency_ms: float,
    ) -> Path:
        safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", request_tag)[:80]
        path = self.raw_directory / f"{safe_tag}-{key[:12]}.json"
        path.write_text(
            json.dumps(
                {
                    "request_payload": request_payload,
                    "cache_key": key,
                    "request_started_utc": started_utc.isoformat(),
                    "request_finished_utc": finished_utc.isoformat(),
                    "physical_latency_ms": physical_latency_ms,
                    "content": content,
                    "provider_payload": provider_payload,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return path.resolve()

    def _write_success_terminal(
        self,
        trace: _TransportTrace,
        *,
        raw_path: Path,
        cache_path: Path | None,
        provider_usage_available: bool,
    ) -> None:
        terminal_path = Path(trace.terminal_path)
        attempt_artifacts = [
            {"path": path, "sha256": _file_sha256(Path(path))}
            for path in trace.attempt_paths
        ]
        value = {
            "schema_version": 1,
            "logical_request_id": trace.logical_request_id,
            "request_tag": trace.request_tag,
            "cache_key": trace.cache_key,
            "attempt_artifacts": attempt_artifacts,
            "transport_attempt_count": len(attempt_artifacts),
            "total_physical_elapsed_ms": trace.physical_elapsed_ms,
            "terminal_outcome": "success",
            "success_raw_path": str(raw_path.resolve()),
            "success_raw_sha256": _file_sha256(raw_path),
            "success_cache_path": str(cache_path) if cache_path is not None else None,
            "success_cache_sha256": (
                _file_sha256(cache_path) if cache_path is not None else None
            ),
            "terminal_error_category": None,
            "provider_usage_available": provider_usage_available,
        }
        _exclusive_json_write(terminal_path, value)


class OpenAICompatibleClient(BaseModelClient):
    """Minimal client for LM Studio and the official OpenAI chat endpoint."""

    def _request(
        self,
        payload: Mapping[str, Any],
        *,
        request_tag: str | None = None,
        cache_key: str | None = None,
    ) -> _RequestResult:
        if not self.config.base_url:
            raise ModelError("base_url is required for an OpenAI-compatible provider")
        request_tag = request_tag or "request"
        cache_key = cache_key or ResponseCache.key(payload)
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get(self.config.api_key_env, "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        maximum_attempts = self.config.max_retries + 1
        logical_request_id = _logical_request_id(request_tag, cache_key)
        request_payload_sha256 = _mapping_sha256(payload)
        attempt_paths: list[str] = []
        physical_elapsed_ms = 0.0
        last_error: Exception | None = None
        last_category = "model_error"
        for attempt in range(maximum_attempts):
            request = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            attempt_started_utc = datetime.now(timezone.utc)
            attempt_started = time.perf_counter()
            response_received = False
            try:
                with urllib.request.urlopen(
                    request, timeout=self.config.timeout_seconds
                ) as response:
                    raw_bytes = response.read()
                    response_received = True
                    raw = raw_bytes.decode("utf-8")
                value = json.loads(raw)
                content, usage = _validated_provider_response(value)
                choice = value["choices"][0]
                message = choice["message"]
                if not content.strip():
                    reasoning = message.get("reasoning_content", message.get("reasoning"))
                    reasoning_length = len(reasoning) if isinstance(reasoning, str) else 0
                    raise _EmptyVisibleContentError(
                        "provider returned empty visible content; hidden reasoning is not an "
                        f"answer (finish_reason={choice.get('finish_reason')!r}, "
                        f"hidden_reasoning_chars={reasoning_length})"
                    )
                if (
                    self.config.reasoning_effort == "none"
                    and _payload_has_hidden_reasoning(value)
                ):
                    raise _HiddenReasoningError(
                        "provider returned hidden reasoning while reasoning_effort='none'"
                    )
                elapsed_ms = (time.perf_counter() - attempt_started) * 1000.0
                physical_elapsed_ms += elapsed_ms
                attempt_path = self._write_attempt_artifact(
                    logical_request_id=logical_request_id,
                    request_tag=request_tag,
                    cache_key=cache_key,
                    attempt_index=attempt + 1,
                    max_attempts=maximum_attempts,
                    started_utc=attempt_started_utc,
                    elapsed_ms=elapsed_ms,
                    endpoint=url,
                    outcome_category="success",
                    exception=None,
                    response_received=True,
                    valid_response_body=True,
                    retry_eligible=False,
                    request_payload_sha256=request_payload_sha256,
                )
                if attempt_path is not None:
                    attempt_paths.append(str(attempt_path))
                trace = self._success_trace(
                    logical_request_id,
                    request_tag,
                    cache_key,
                    attempt_paths,
                    physical_elapsed_ms,
                )
                return _RequestResult(content, usage, value, trace)
            except Exception as exc:
                last_error = exc
                (
                    category,
                    retryable,
                    classified_received,
                    valid_response_body,
                ) = _classify_request_error(exc, response_received=response_received)
                response_received = classified_received
                last_category = category
                elapsed_ms = (time.perf_counter() - attempt_started) * 1000.0
                physical_elapsed_ms += elapsed_ms
                attempt_path = self._write_attempt_artifact(
                    logical_request_id=logical_request_id,
                    request_tag=request_tag,
                    cache_key=cache_key,
                    attempt_index=attempt + 1,
                    max_attempts=maximum_attempts,
                    started_utc=attempt_started_utc,
                    elapsed_ms=elapsed_ms,
                    endpoint=url,
                    outcome_category=category,
                    exception=exc,
                    response_received=response_received,
                    valid_response_body=valid_response_body,
                    retry_eligible=retryable,
                    request_payload_sha256=request_payload_sha256,
                )
                if attempt_path is not None:
                    attempt_paths.append(str(attempt_path))
                if retryable and attempt + 1 < maximum_attempts:
                    time.sleep(self.config.retry_backoff_seconds)
                    continue
                break
        if self.terminal_directory is None:
            raise ModelError(
                f"model request failed after {maximum_attempts} attempt(s): {last_error}"
            )
        terminal_path = self._write_failure_terminal(
            logical_request_id=logical_request_id,
            request_tag=request_tag,
            cache_key=cache_key,
            attempt_paths=attempt_paths,
            physical_elapsed_ms=physical_elapsed_ms,
            error_category=last_category,
        )
        raise ModelRequestError(
            f"model request failed after {len(attempt_paths)} attempt(s): {last_error}",
            logical_request_id=logical_request_id,
            transport_attempt_paths=attempt_paths,
            terminal_evidence_path=str(terminal_path),
            physical_elapsed_ms=physical_elapsed_ms,
            error_category=last_category,
        )

    def _write_attempt_artifact(
        self,
        *,
        logical_request_id: str,
        request_tag: str,
        cache_key: str,
        attempt_index: int,
        max_attempts: int,
        started_utc: datetime,
        elapsed_ms: float,
        endpoint: str,
        outcome_category: str,
        exception: Exception | None,
        response_received: bool,
        valid_response_body: bool,
        retry_eligible: bool,
        request_payload_sha256: str,
    ) -> Path | None:
        if self.attempts_directory is None:
            return None
        finished_utc = datetime.now(timezone.utc)
        path = (
            self.attempts_directory
            / logical_request_id
            / f"attempt-{attempt_index:02d}.json"
        ).resolve()
        value = {
            "schema_version": 1,
            "logical_request_id": logical_request_id,
            "request_tag": request_tag,
            "cache_key": cache_key,
            "attempt_index": attempt_index,
            "max_attempts": max_attempts,
            "started_utc": started_utc.isoformat(),
            "finished_utc": finished_utc.isoformat(),
            "elapsed_ms": elapsed_ms,
            "provider": self.config.provider,
            "model": self.config.model_id,
            "endpoint": endpoint,
            "timeout_seconds": self.config.timeout_seconds,
            "max_tokens": self.config.max_tokens,
            "outcome_category": outcome_category,
            "exception_class": type(exception).__name__ if exception else None,
            "exception_message": str(exception) if exception else None,
            "response_received": response_received,
            "valid_response_body": valid_response_body,
            "retry_eligible": retry_eligible,
            "request_payload_sha256": request_payload_sha256,
        }
        _exclusive_json_write(path, value)
        return path

    def _success_trace(
        self,
        logical_request_id: str,
        request_tag: str,
        cache_key: str,
        attempt_paths: Sequence[str],
        physical_elapsed_ms: float,
    ) -> _TransportTrace | None:
        if self.terminal_directory is None:
            return None
        return _TransportTrace(
            logical_request_id=logical_request_id,
            request_tag=request_tag,
            cache_key=cache_key,
            attempt_paths=tuple(attempt_paths),
            physical_elapsed_ms=physical_elapsed_ms,
            terminal_path=str(
                (self.terminal_directory / f"{logical_request_id}.json").resolve()
            ),
        )

    def _write_failure_terminal(
        self,
        *,
        logical_request_id: str,
        request_tag: str,
        cache_key: str,
        attempt_paths: Sequence[str],
        physical_elapsed_ms: float,
        error_category: str,
    ) -> Path:
        if self.terminal_directory is None:
            raise ModelError("classified retries require an append-only terminal_dir")
        path = (self.terminal_directory / f"{logical_request_id}.json").resolve()
        attempt_artifacts = [
            {"path": item, "sha256": _file_sha256(Path(item))}
            for item in attempt_paths
        ]
        value = {
            "schema_version": 1,
            "logical_request_id": logical_request_id,
            "request_tag": request_tag,
            "cache_key": cache_key,
            "attempt_artifacts": attempt_artifacts,
            "transport_attempt_count": len(attempt_artifacts),
            "total_physical_elapsed_ms": physical_elapsed_ms,
            "terminal_outcome": "failure",
            "success_raw_path": None,
            "success_raw_sha256": None,
            "success_cache_path": None,
            "success_cache_sha256": None,
            "terminal_error_category": error_category,
            "provider_usage_available": False,
        }
        _exclusive_json_write(path, value)
        return path


class MockModelClient(BaseModelClient):
    """Deterministic provider that understands only the repository fixtures."""

    def _request(
        self,
        payload: Mapping[str, Any],
        *,
        request_tag: str | None = None,
        cache_key: str | None = None,
    ) -> _RequestResult:
        messages = payload["messages"]
        system_text = "\n".join(
            message["content"] for message in messages if message["role"] == "system"
        )
        user_text = "\n".join(
            message["content"] for message in messages if message["role"] == "user"
        )
        safe_input = _extract_numbered_input(user_text)
        theory = safe_input["theory"]
        question = safe_input["question"]
        formalization = formalize_controlled(theory, question)
        serialized = formalization_to_v2_dict(formalization)
        mode_match = re.search(r"MODE=([a-z_]+)", system_text)
        mode = mode_match.group(1) if mode_match else "constrained"
        is_repair = "REPAIR_REQUEST" in system_text

        if mode in {"direct", "cot", "cot_refine"}:
            result = ForwardChainingSolver().solve(formalization)
            if mode == "direct":
                content = result.label
            elif mode == "cot_refine":
                content = (
                    "REVIEW:\n"
                    "I checked the initial reasoning against the original question "
                    "using forward rules and open-world semantics.\n\n"
                    f"FINAL_LABEL_FOR_ORIGINAL_QUESTION: {result.label}"
                )
            else:
                content = (
                    "REASONING:\n"
                    "I applied the stated rules forward under open-world semantics "
                    "to the original question.\n\n"
                    f"FINAL_LABEL_FOR_ORIGINAL_QUESTION: {result.label}"
                )
        elif mode == "plain":
            content = "```json\n" + json.dumps(serialized) + "\n```"
        else:
            if mode == "vgcf_legacy" and not is_repair and _should_inject_error(safe_input):
                if serialized["facts"]:
                    serialized["facts"][0].pop("id", None)
                elif serialized["rules"]:
                    serialized["rules"][0]["id"] = "s999"
            content = json.dumps(serialized, ensure_ascii=False, separators=(",", ":"))
        usage = {
            "prompt_tokens": _estimate_tokens(messages),
            "completion_tokens": max(1, len(content) // 4),
        }
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        provider_payload = {"mock": True, "mode": mode, "repaired": is_repair}
        return _RequestResult(content, usage, provider_payload)


def _extract_json_after_marker(text: str, marker: str) -> Mapping[str, Any]:
    marker_index = text.find(marker)
    if marker_index < 0:
        raise ModelError(f"Mock request missing {marker}")
    object_index = text.find("{", marker_index + len(marker))
    if object_index < 0:
        raise ModelError(f"Mock request has no JSON after {marker}")
    try:
        value, _ = json.JSONDecoder().raw_decode(text[object_index:])
    except json.JSONDecodeError as exc:
        raise ModelError(f"Mock cannot decode safe input: {exc}") from exc
    if not isinstance(value, dict) or set(value) != {"theory", "question"}:
        raise ModelError("Mock safe input violates the field whitelist")
    return value


def _extract_numbered_input(text: str) -> Mapping[str, Any]:
    marker = "NUMBERED_INPUT"
    marker_index = text.find(marker)
    if marker_index < 0:
        raise ModelError(f"Mock request missing {marker}")
    section = text[marker_index + len(marker) :]
    theory_by_index: dict[int, str] = {}
    question: str | None = None
    for line in section.splitlines():
        source = re.fullmatch(r"s([1-9][0-9]*): (.+)", line)
        if source:
            theory_by_index[int(source.group(1))] = source.group(2)
            continue
        query = re.fullmatch(r"q1: (.+)", line)
        if query:
            question = query.group(1)
            continue
        if line in {
            "ORIGINAL_FORMALIZATION",
            "VALIDATION_ISSUES",
            "HARD_ISSUES_MANDATORY",
            "SOFT_WARNINGS_ADVISORY_NON_MANDATORY",
            "INITIAL_COT",
        }:
            break
    expected = list(range(1, len(theory_by_index) + 1))
    if sorted(theory_by_index) != expected or question is None:
        raise ModelError("Mock numbered input must contain contiguous s1..sN and q1")
    return {
        "theory": [theory_by_index[index] for index in expected],
        "question": question,
    }


def _should_inject_error(safe_input: Mapping[str, Any]) -> bool:
    stable = json.dumps(safe_input, sort_keys=True, ensure_ascii=False)
    return int(hashlib.sha256(stable.encode("utf-8")).hexdigest()[:8], 16) % 4 == 0


def _estimate_tokens(messages: Sequence[Mapping[str, str]]) -> int:
    characters = sum(len(message.get("content", "")) for message in messages)
    return max(1, characters // 4)


def _has_hidden_reasoning(response: ModelResponse) -> bool:
    return _payload_has_hidden_reasoning(response.provider_payload)


def _payload_has_hidden_reasoning(payload: Mapping[str, Any]) -> bool:
    try:
        message = payload["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return False
    if not isinstance(message, Mapping):
        return False
    reasoning = message.get("reasoning_content", message.get("reasoning"))
    return isinstance(reasoning, str) and bool(reasoning.strip())


def _validated_provider_response(
    value: Any,
) -> tuple[str, Mapping[str, Any]]:
    if not isinstance(value, Mapping):
        raise _ProviderSchemaError("provider response must be a JSON object")
    choices = value.get("choices")
    if not isinstance(choices, list) or not choices:
        raise _ProviderSchemaError("provider response choices must be a non-empty list")
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise _ProviderSchemaError("provider response choice must be an object")
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise _ProviderSchemaError("provider response message must be an object")
    content = message.get("content")
    if not isinstance(content, str):
        raise _ProviderSchemaError("provider returned a non-string message content")
    usage = value.get("usage") or {}
    if not isinstance(usage, Mapping):
        raise _ProviderSchemaError("provider usage must be an object when present")
    return content, usage


def _coerce_request_result(value: Any) -> _RequestResult:
    if isinstance(value, _RequestResult):
        return value
    if isinstance(value, tuple) and len(value) == 3:
        content, usage, provider_payload = value
        if (
            isinstance(content, str)
            and isinstance(usage, Mapping)
            and isinstance(provider_payload, Mapping)
        ):
            return _RequestResult(content, usage, provider_payload)
    raise ModelError("model client returned an invalid request result")


def _classify_request_error(
    error: Exception, *, response_received: bool
) -> tuple[str, bool, bool, bool]:
    if isinstance(error, urllib.error.HTTPError):
        body = b""
        try:
            body = error.read()
        except OSError:
            body = b""
        received = bool(body)
        valid_body = False
        if body:
            try:
                decoded = json.loads(body.decode("utf-8"))
                content, _ = _validated_provider_response(decoded)
                valid_body = bool(content.strip())
            except (
                UnicodeDecodeError,
                json.JSONDecodeError,
                _ProviderSchemaError,
            ):
                valid_body = False
        category = f"http_{error.code}"
        retryable = error.code in {429, 502, 503, 504} and not valid_body
        return category, retryable, received, valid_body
    root: BaseException = error
    if isinstance(error, urllib.error.URLError) and isinstance(
        error.reason, BaseException
    ):
        root = error.reason
    if isinstance(root, ConnectionRefusedError):
        return "connection_refused", True, response_received, False
    if isinstance(root, ConnectionResetError):
        return "connection_reset", True, response_received, False
    if isinstance(root, TimeoutError):
        return "timeout", False, response_received, False
    if isinstance(error, (json.JSONDecodeError, UnicodeDecodeError)):
        return "json_decode_error", False, True, False
    if isinstance(error, _ProviderSchemaError):
        return "provider_schema_error", False, response_received, True
    if isinstance(error, _EmptyVisibleContentError):
        return "empty_visible_content", False, response_received, True
    if isinstance(error, _HiddenReasoningError):
        return "hidden_reasoning_contamination", False, response_received, True
    if isinstance(error, ModelError):
        return "model_error", False, response_received, response_received
    if isinstance(error, urllib.error.URLError):
        return "transport_error", False, response_received, False
    return "unexpected_error", False, response_received, response_received


def _logical_request_id(request_tag: str, cache_key: str) -> str:
    safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", request_tag).strip("._-")[:48]
    prefix = safe_tag or "request"
    identity = hashlib.sha256(
        f"{request_tag}\0{cache_key}".encode("utf-8")
    ).hexdigest()[:16]
    return f"{prefix}-{identity}"


def _mapping_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _exclusive_json_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _usage_available(usage: Mapping[str, Any]) -> bool:
    return any(
        isinstance(usage.get(field), (int, float))
        and not isinstance(usage.get(field), bool)
        for field in ("prompt_tokens", "completion_tokens", "total_tokens")
    )


def create_client(config: ClientConfig) -> BaseModelClient:
    if config.provider == "mock":
        return MockModelClient(config)
    if config.provider in {"lmstudio", "openai", "openai_compatible"}:
        return OpenAICompatibleClient(config)
    raise ModelError(f"unsupported provider: {config.provider}")
