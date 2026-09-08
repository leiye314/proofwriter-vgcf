"""Fair VGCF-2 method families sharing one constrained initial generation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .data import ModelExample, build_model_input, render_model_input
from .errors import VGCFError
from .model import BaseModelClient, ModelResponse
from .parsing import parse_cot_label, parse_cot_refine_label, parse_label
from .protocol import audit_model_example
from .schema import Formalization
from .solver import ForwardChainingSolver, SolveResult
from .validator import (
    StaticValidator,
    ValidationResult,
    issue_dicts,
    merge_repair_issues,
)

METHODS = (
    "direct",
    "cot",
    "cot_refine",
    "plain",
    "constrained",
    "gate_cot",
    "repair_only",
    "vgcf2",
    "vgcf",  # legacy spelling; executes the VGCF-2 route
)
FORMALIZATION_METHODS = frozenset(
    {"plain", "constrained", "gate_cot", "repair_only", "vgcf2", "vgcf"}
)


@dataclass(frozen=True, slots=True)
class MethodOutput:
    method: str
    predicted_label: str
    raw_model_response: str
    parsed_structure: Mapping[str, Any] | None
    validation_errors: tuple[Mapping[str, Any], ...]
    repair_triggered: bool
    repair_attempted: bool
    repair_improved: bool
    repair_regression: bool
    repair_regression_layers: tuple[str, ...]
    pre_repair_result: Mapping[str, Any] | None
    post_repair_result: Mapping[str, Any] | None
    proof_trace: tuple[Mapping[str, Any], ...]
    json_parse_valid: bool | None
    strict_schema_valid: bool | None
    normalized_schema_valid: bool | None
    schema_valid: bool | None
    hard_validator_pass: bool | None
    soft_issue_count: int | None
    solver_executable: bool | None
    strict_solver_executable: bool | None
    normalized_solver_executable: bool | None
    singleton_if_normalized_count: int | None
    shadow_solver_label: str | None
    answer_from_solver: bool
    route_source: str
    fallback_used: bool
    coverage: bool
    infrastructure_error: bool
    method_error: bool
    cot_refine_contract_valid: bool | None
    call_count: int
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    model_id: str
    prompt_hash: str
    cached: bool
    raw_output_paths: tuple[str, ...]
    initial_response_hash: str | None
    initial_raw_output_path: str | None
    initial_cache_key: str | None
    answer_response_hash: str | None
    answer_raw_output_path: str | None
    answer_cache_key: str | None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class _Stage:
    raw: str
    validation: ValidationResult
    solve_result: SolveResult | None
    solver_error: str | None
    strict_solve_result: SolveResult | None
    strict_solver_error: str | None

    @property
    def solver_executable(self) -> bool:
        return self.solve_result is not None

    @property
    def strict_solver_executable(self) -> bool:
        return self.strict_solve_result is not None


class MethodRunner:
    """Run model methods without accepting labels, gold programs, or data paths."""

    def __init__(
        self,
        client: BaseModelClient,
        prompt_directory: str | Path,
        solver: ForwardChainingSolver | None = None,
        validator: StaticValidator | None = None,
        *,
        phase: str = "software_validation",
        final_test_armed: bool = False,
    ) -> None:
        self.client = client
        self.prompt_directory = Path(prompt_directory)
        self.solver = solver or ForwardChainingSolver()
        self.validator = validator or StaticValidator()
        self.phase = phase
        self.final_test_armed = final_test_armed
        self._initial_responses: dict[str, ModelResponse] = {}
        self._cot_responses: dict[str, ModelResponse] = {}

    def run(self, method: str, example: ModelExample) -> MethodOutput:
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
        if not isinstance(example, ModelExample):
            raise TypeError("MethodRunner accepts only ModelExample without evaluator gold")
        audit_model_example(
            self.phase, example, final_test_armed=self.final_test_armed
        )
        safe_input = build_model_input(example)
        if method in {"direct", "cot"}:
            return self._run_label(method, example, safe_input)
        if method == "cot_refine":
            return self._run_cot_refine(example, safe_input)
        if method == "plain":
            return self._run_plain(example, safe_input)

        try:
            initial = self._constrained_initial(example, safe_input)
        except VGCFError as exc:
            return self._request_error_output(method, self._load_prompt("constrained"), exc)
        stage = self._evaluate(initial.content, example)
        if method == "constrained":
            return self._constrained_output(method, initial, stage)
        if method == "gate_cot":
            if stage.validation.hard_valid and stage.solver_executable:
                return self._stage_output(
                    method, [initial], stage, "solver_initial", call_count=1
                )
            return self._cot_fallback(
                method, example, safe_input, initial, stage, None, None, None
            )
        if method == "repair_only":
            return self._repair_only(method, example, safe_input, initial, stage)
        return self._run_vgcf2(method, example, safe_input, initial, stage)

    def _run_label(
        self,
        method: str,
        example: ModelExample,
        safe_input: Mapping[str, Any],
    ) -> MethodOutput:
        prompt = self._load_prompt(method)
        try:
            response = self.client.complete(
                self._messages(prompt, safe_input),
                request_tag=f"{method}-{example.example_id}",
            )
        except VGCFError as exc:
            return self._request_error_output(method, prompt, exc)
        return self._label_output(method, [prompt], [response], call_count=1)

    def _run_cot_refine(
        self, example: ModelExample, safe_input: Mapping[str, Any]
    ) -> MethodOutput:
        cot_prompt = self._load_prompt("cot")
        refine_prompt = self._load_prompt("cot_refine")
        try:
            first = self._cot_response(example, safe_input)
        except VGCFError as exc:
            return self._request_error_output(
                "cot_refine", cot_prompt + "\n" + refine_prompt, exc, call_count=1
            )
        try:
            second = self.client.complete(
                self._refine_messages(refine_prompt, safe_input, first.content),
                request_tag=f"cot-refine-{example.example_id}",
            )
        except VGCFError as exc:
            return self._request_error_output(
                "cot_refine", cot_prompt + "\n" + refine_prompt, exc, call_count=2
            )
        return self._label_output(
            "cot_refine", [cot_prompt, refine_prompt], [first, second], call_count=2
        )

    def _run_plain(
        self, example: ModelExample, safe_input: Mapping[str, Any]
    ) -> MethodOutput:
        prompt = self._load_prompt("plain")
        try:
            response = self.client.complete(
                self._messages(prompt, safe_input),
                request_tag=f"plain-{example.example_id}",
            )
        except VGCFError as exc:
            return self._request_error_output("plain", prompt, exc)
        stage = self._evaluate(response.content, example)
        return self._constrained_output("plain", response, stage, prompt_name="plain")

    def _constrained_initial(
        self, example: ModelExample, safe_input: Mapping[str, Any]
    ) -> ModelResponse:
        key = _safe_input_hash(safe_input)
        if key not in self._initial_responses:
            prompt = self._load_prompt("constrained")
            self._initial_responses[key] = self.client.complete(
                self._messages(prompt, safe_input),
                request_tag=f"constrained-base-{example.example_id}",
            )
        return self._initial_responses[key]

    def _cot_response(
        self, example: ModelExample, safe_input: Mapping[str, Any]
    ) -> ModelResponse:
        key = _safe_input_hash(safe_input)
        if key not in self._cot_responses:
            prompt = self._load_prompt("cot")
            self._cot_responses[key] = self.client.complete(
                self._messages(prompt, safe_input),
                request_tag=f"cot-{example.example_id}",
            )
        return self._cot_responses[key]

    def _evaluate(self, raw: str, example: ModelExample) -> _Stage:
        validation = self.validator.validate_json(raw, example)
        solved: SolveResult | None = None
        solver_error: str | None = None
        if validation.formalization is not None:
            try:
                solved = self.solver.solve(validation.formalization)
            except VGCFError as exc:
                solver_error = str(exc)
        strict_solved: SolveResult | None = None
        strict_solver_error: str | None = None
        if validation.strict_formalization is not None:
            if validation.strict_formalization == validation.formalization:
                strict_solved = solved
                strict_solver_error = solver_error
            else:
                try:
                    strict_solved = self.solver.solve(validation.strict_formalization)
                except VGCFError as exc:
                    strict_solver_error = str(exc)
        return _Stage(
            raw,
            validation,
            solved,
            solver_error,
            strict_solved,
            strict_solver_error,
        )

    def _constrained_output(
        self,
        method: str,
        response: ModelResponse,
        stage: _Stage,
        *,
        prompt_name: str = "constrained",
    ) -> MethodOutput:
        if stage.solver_executable:
            return self._stage_output(
                method,
                [response],
                stage,
                "solver_initial",
                call_count=1,
                prompt_names=[prompt_name],
            )
        error = stage.solver_error or "initial formalization is not solver-executable"
        return self._stage_output(
            method,
            [response],
            stage,
            "method_error",
            call_count=1,
            prompt_names=[prompt_name],
            predicted_label="Error",
            method_error=True,
            error=error,
        )

    def _attempt_repair(
        self,
        example: ModelExample,
        safe_input: Mapping[str, Any],
        initial: _Stage,
    ) -> tuple[ModelResponse | None, _Stage | None, str | None]:
        prompt = self._load_prompt("repair")
        hard_issues = merge_repair_issues(
            initial.validation.issues, limit=5, severity="hard"
        )
        if initial.solver_error and len(hard_issues) < 5:
            hard_issues.append(
                {
                    "code": "solver_execution_error",
                    "severity": "hard",
                    "message": initial.solver_error,
                    "paths": ["$"],
                    "repair_hint": "Return a schema-valid program executable by the documented solver.",
                    "count": 1,
                }
            )
        advisories = merge_repair_issues(
            initial.validation.issues, limit=3, severity="soft"
        )
        try:
            response = self.client.complete(
                self._repair_messages(
                    prompt, safe_input, initial.raw, hard_issues, advisories
                ),
                request_tag=f"vgcf2-repair-{example.example_id}",
            )
        except VGCFError as exc:
            return None, None, str(exc)
        return response, self._evaluate(response.content, example), None

    @staticmethod
    def _choose_stage(initial: _Stage, repaired: _Stage) -> tuple[_Stage, bool]:
        initial_rank = (
            int(initial.solver_executable),
            int(initial.validation.hard_valid),
            -len(initial.validation.hard_issues),
            -len(initial.validation.soft_issues),
        )
        repaired_rank = (
            int(repaired.solver_executable),
            int(repaired.validation.hard_valid),
            -len(repaired.validation.hard_issues),
            -len(repaired.validation.soft_issues),
        )
        if repaired_rank > initial_rank:
            return repaired, True
        return initial, False

    def _repair_only(
        self,
        method: str,
        example: ModelExample,
        safe_input: Mapping[str, Any],
        initial_response: ModelResponse,
        initial: _Stage,
    ) -> MethodOutput:
        pre = self._snapshot(initial)
        if initial.validation.hard_valid and initial.solver_executable:
            return self._stage_output(
                method,
                [initial_response],
                initial,
                "solver_initial",
                call_count=1,
                pre=pre,
            )
        repair_response, repaired, repair_error = self._attempt_repair(
            example, safe_input, initial
        )
        if repair_response is None or repaired is None:
            if initial.solver_executable:
                return self._stage_output(
                    method,
                    [initial_response],
                    initial,
                    "solver_initial",
                    call_count=2,
                    pre=pre,
                    repair_attempted=True,
                    infrastructure_error=True,
                    error=f"repair request failed; preserved executable initial: {repair_error}",
                )
            return self._stage_output(
                method,
                [initial_response],
                initial,
                "infrastructure_error",
                call_count=2,
                pre=pre,
                repair_attempted=True,
                predicted_label="Error",
                infrastructure_error=True,
                error=f"repair request failed: {repair_error}",
            )
        selected, improved = self._choose_stage(initial, repaired)
        responses = [initial_response, repair_response]
        post = self._snapshot(repaired)
        if selected.solver_executable:
            route = "solver_repair" if selected is repaired else "solver_initial"
            return self._stage_output(
                method,
                responses,
                selected,
                route,
                call_count=2,
                pre=pre,
                post=post,
                repair_attempted=True,
                repair_improved=improved,
                all_issues=initial.validation.issues + repaired.validation.issues,
            )
        return self._stage_output(
            method,
            responses,
            selected,
            "method_error",
            call_count=2,
            pre=pre,
            post=post,
            repair_attempted=True,
            repair_improved=improved,
            all_issues=initial.validation.issues + repaired.validation.issues,
            predicted_label="Error",
            method_error=True,
            error="repair did not produce an executable formalization",
        )

    def _run_vgcf2(
        self,
        method: str,
        example: ModelExample,
        safe_input: Mapping[str, Any],
        initial_response: ModelResponse,
        initial: _Stage,
    ) -> MethodOutput:
        pre = self._snapshot(initial)
        if initial.validation.hard_valid and initial.solver_executable:
            return self._stage_output(
                method,
                [initial_response],
                initial,
                "solver_initial",
                call_count=1,
                pre=pre,
            )
        repair_response, repaired, repair_error = self._attempt_repair(
            example, safe_input, initial
        )
        if repair_response is not None and repaired is not None:
            selected, improved = self._choose_stage(initial, repaired)
            post = self._snapshot(repaired)
            if repaired.validation.hard_valid and repaired.solver_executable:
                return self._stage_output(
                    method,
                    [initial_response, repair_response],
                    repaired,
                    "solver_repair",
                    call_count=2,
                    pre=pre,
                    post=post,
                    repair_attempted=True,
                    repair_improved=improved,
                    all_issues=initial.validation.issues + repaired.validation.issues,
                )
        else:
            selected, improved, post = initial, False, None
        return self._cot_fallback(
            method,
            example,
            safe_input,
            initial_response,
            selected,
            repair_response,
            post,
            repair_error,
            repair_attempted=True,
            repair_improved=improved,
            pre=pre,
            repaired_stage=repaired,
        )

    def _cot_fallback(
        self,
        method: str,
        example: ModelExample,
        safe_input: Mapping[str, Any],
        initial_response: ModelResponse,
        selected: _Stage,
        repair_response: ModelResponse | None,
        post: Mapping[str, Any] | None,
        repair_error: str | None,
        *,
        repair_attempted: bool = False,
        repair_improved: bool = False,
        pre: Mapping[str, Any] | None = None,
        repaired_stage: _Stage | None = None,
    ) -> MethodOutput:
        responses = [initial_response]
        if repair_response is not None:
            responses.append(repair_response)
        try:
            cot = self._cot_response(example, safe_input)
        except VGCFError as exc:
            return self._stage_output(
                method,
                responses,
                selected,
                "infrastructure_error",
                call_count=1 + int(repair_attempted) + 1,
                pre=pre,
                post=post,
                repair_attempted=repair_attempted,
                repair_improved=repair_improved,
                predicted_label="Error",
                fallback_used=True,
                infrastructure_error=True,
                error=f"CoT fallback failed: {exc}",
            )
        responses.append(cot)
        try:
            label = parse_cot_label(cot.content)
        except VGCFError as exc:
            return self._stage_output(
                method,
                responses,
                selected,
                "method_error",
                call_count=1 + int(repair_attempted) + 1,
                pre=pre,
                post=post,
                repair_attempted=repair_attempted,
                repair_improved=repair_improved,
                predicted_label="Error",
                fallback_used=True,
                infrastructure_error=repair_error is not None,
                method_error=True,
                error=f"CoT fallback output contract failed: {exc}",
                answer_response=cot,
                prompt_names=(
                    ["constrained", "repair", "cot"]
                    if repair_attempted
                    else ["constrained", "cot"]
                ),
            )
        issues = selected.validation.issues
        if repaired_stage is not None:
            issues = selected.validation.issues + repaired_stage.validation.issues
        error = f"repair request failed before fallback: {repair_error}" if repair_error else None
        return self._stage_output(
            method,
            responses,
            selected,
            "cot_fallback",
            call_count=1 + int(repair_attempted) + 1,
            pre=pre,
            post=post,
            repair_attempted=repair_attempted,
            repair_improved=repair_improved,
            all_issues=issues,
            predicted_label=label,
            fallback_used=True,
            infrastructure_error=repair_error is not None,
            error=error,
            answer_response=cot,
            prompt_names=(
                ["constrained", "repair", "cot"]
                if repair_attempted
                else ["constrained", "cot"]
            ),
        )

    @staticmethod
    def _messages(prompt: str, safe_input: Mapping[str, Any]) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": "NUMBERED_INPUT\n" + render_model_input(safe_input),
            },
        ]

    @staticmethod
    def _repair_messages(
        repair_prompt: str,
        safe_input: Mapping[str, Any],
        previous_output: str,
        hard_issues: Sequence[Mapping[str, Any]],
        advisories: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": repair_prompt},
            {
                "role": "user",
                "content": (
                    "NUMBERED_INPUT\n"
                    + render_model_input(safe_input)
                    + "\nORIGINAL_FORMALIZATION\n"
                    + previous_output
                    + "\nHARD_ISSUES_MANDATORY\n"
                    + json.dumps(
                        list(hard_issues),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\nSOFT_WARNINGS_ADVISORY_NON_MANDATORY\n"
                    + json.dumps(
                        list(advisories),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                ),
            },
        ]

    @staticmethod
    def _refine_messages(
        prompt: str, safe_input: Mapping[str, Any], initial_cot: str
    ) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    "NUMBERED_INPUT\n"
                    + render_model_input(safe_input)
                    + "\nINITIAL_COT\n"
                    + initial_cot
                ),
            },
        ]

    def _load_prompt(self, method: str) -> str:
        path = self.prompt_directory / f"{method}.txt"
        if not path.exists():
            raise FileNotFoundError(f"prompt file not found: {path}")
        return path.read_text(encoding="utf-8").strip()

    def _snapshot(self, stage: _Stage) -> dict[str, Any]:
        return {
            "raw_model_response": stage.raw,
            "json_parse_valid": stage.validation.json_parse_valid,
            "strict_schema_valid": stage.validation.strict_schema_valid,
            "normalized_schema_valid": stage.validation.normalized_schema_valid,
            "schema_valid": stage.validation.schema_valid,
            "hard_validator_pass": stage.validation.hard_valid,
            "soft_issue_count": len(stage.validation.soft_issues),
            "trust_score": stage.validation.trust_score,
            "solver_executable": stage.solver_executable,
            "strict_solver_executable": stage.strict_solver_executable,
            "normalized_solver_executable": stage.solver_executable,
            "singleton_if_normalized_count": (
                stage.validation.singleton_if_normalized_count
            ),
            "solver_label": (
                stage.solve_result.label if stage.solve_result is not None else None
            ),
            "parsed_structure": (
                stage.validation.formalization.to_dict()
                if stage.validation.formalization is not None
                else None
            ),
            "validation_errors": issue_dicts(stage.validation.issues),
            "solver_error": stage.solver_error,
            "strict_solver_error": stage.strict_solver_error,
        }

    def _stage_output(
        self,
        method: str,
        responses: Sequence[ModelResponse],
        stage: _Stage,
        route_source: str,
        *,
        call_count: int,
        prompt_names: Sequence[str] | None = None,
        pre: Mapping[str, Any] | None = None,
        post: Mapping[str, Any] | None = None,
        repair_attempted: bool = False,
        repair_improved: bool = False,
        all_issues: Sequence[Any] | None = None,
        predicted_label: str | None = None,
        fallback_used: bool = False,
        infrastructure_error: bool = False,
        method_error: bool = False,
        error: str | None = None,
        answer_response: ModelResponse | None = None,
    ) -> MethodOutput:
        combined = _combine_responses(*responses)
        formalization = stage.validation.formalization
        label = predicted_label or (
            stage.solve_result.label if stage.solve_result is not None else "Error"
        )
        proof = (
            tuple(step.to_dict() for step in stage.solve_result.proof)
            if predicted_label is None and stage.solve_result is not None
            else ()
        )
        if prompt_names is None:
            prompt_names = ["constrained"]
            if repair_attempted:
                prompt_names.append("repair")
            if fallback_used:
                prompt_names.append("cot")
        raw = (
            responses[0].content
            if len(responses) == 1
            else json.dumps(
                {
                    "initial": responses[0].content,
                    **(
                        {"repair": responses[1].content}
                        if repair_attempted and post is not None and len(responses) >= 2
                        else {}
                    ),
                    **({"fallback": responses[-1].content} if fallback_used else {}),
                },
                ensure_ascii=False,
            )
        )
        issues = all_issues if all_issues is not None else stage.validation.issues
        regression_layers = (
            _repair_regression_layers(pre, post) if repair_attempted else ()
        )
        return MethodOutput(
            method=method,
            predicted_label=label,
            raw_model_response=raw,
            parsed_structure=formalization.to_dict() if formalization else None,
            validation_errors=tuple(issue_dicts(issues)),
            repair_triggered=repair_attempted,
            repair_attempted=repair_attempted,
            repair_improved=repair_improved,
            repair_regression=bool(regression_layers),
            repair_regression_layers=regression_layers,
            pre_repair_result=pre,
            post_repair_result=post,
            proof_trace=proof,
            json_parse_valid=stage.validation.json_parse_valid,
            strict_schema_valid=stage.validation.strict_schema_valid,
            normalized_schema_valid=stage.validation.normalized_schema_valid,
            schema_valid=stage.validation.schema_valid,
            hard_validator_pass=stage.validation.hard_valid,
            soft_issue_count=len(stage.validation.soft_issues),
            solver_executable=stage.solver_executable,
            strict_solver_executable=stage.strict_solver_executable,
            normalized_solver_executable=stage.solver_executable,
            singleton_if_normalized_count=(
                stage.validation.singleton_if_normalized_count
            ),
            shadow_solver_label=(
                stage.solve_result.label if stage.solve_result is not None else None
            ),
            answer_from_solver=route_source in {"solver_initial", "solver_repair"},
            route_source=route_source,
            fallback_used=fallback_used,
            coverage=label in {"True", "False", "Unknown"},
            infrastructure_error=infrastructure_error,
            method_error=method_error,
            cot_refine_contract_valid=None,
            call_count=call_count,
            latency_ms=combined.latency_ms,
            prompt_tokens=combined.prompt_tokens,
            completion_tokens=combined.completion_tokens,
            total_tokens=combined.total_tokens,
            cost_usd=combined.cost_usd,
            model_id=combined.model_id,
            prompt_hash=_hash_prompt(
                "\n".join(self._load_prompt(name) for name in prompt_names)
            ),
            cached=combined.cached,
            raw_output_paths=combined.raw_output_paths,
            initial_response_hash=_response_hash(responses[0]),
            initial_raw_output_path=responses[0].raw_output_path,
            initial_cache_key=responses[0].cache_key,
            answer_response_hash=(
                _response_hash(answer_response) if answer_response is not None else None
            ),
            answer_raw_output_path=(
                answer_response.raw_output_path if answer_response is not None else None
            ),
            answer_cache_key=(
                answer_response.cache_key if answer_response is not None else None
            ),
            error=error,
        )

    def _label_output(
        self,
        method: str,
        prompts: Sequence[str],
        responses: Sequence[ModelResponse],
        *,
        call_count: int,
    ) -> MethodOutput:
        combined = _combine_responses(*responses)
        parser = {
            "cot": parse_cot_label,
            "cot_refine": parse_cot_refine_label,
        }.get(method, parse_label)
        try:
            label = parser(responses[-1].content)
        except VGCFError as exc:
            label, error, method_error = "Error", str(exc), True
        else:
            error, method_error = None, False
        return MethodOutput(
            method=method,
            predicted_label=label,
            raw_model_response=(
                responses[-1].content
                if len(responses) == 1
                else json.dumps(
                    {"initial": responses[0].content, "refined": responses[-1].content},
                    ensure_ascii=False,
                )
            ),
            parsed_structure=None,
            validation_errors=(),
            repair_triggered=False,
            repair_attempted=False,
            repair_improved=False,
            repair_regression=False,
            repair_regression_layers=(),
            pre_repair_result=None,
            post_repair_result=None,
            proof_trace=(),
            json_parse_valid=None,
            strict_schema_valid=None,
            normalized_schema_valid=None,
            schema_valid=None,
            hard_validator_pass=None,
            soft_issue_count=None,
            solver_executable=None,
            strict_solver_executable=None,
            normalized_solver_executable=None,
            singleton_if_normalized_count=None,
            shadow_solver_label=None,
            answer_from_solver=False,
            route_source="cot_refine" if method == "cot_refine" else method,
            fallback_used=False,
            coverage=label in {"True", "False", "Unknown"},
            infrastructure_error=False,
            method_error=method_error,
            cot_refine_contract_valid=(not method_error)
            if method == "cot_refine"
            else None,
            call_count=call_count,
            latency_ms=combined.latency_ms,
            prompt_tokens=combined.prompt_tokens,
            completion_tokens=combined.completion_tokens,
            total_tokens=combined.total_tokens,
            cost_usd=combined.cost_usd,
            model_id=combined.model_id,
            prompt_hash=_hash_prompt("\n".join(prompts)),
            cached=combined.cached,
            raw_output_paths=combined.raw_output_paths,
            initial_response_hash=None,
            initial_raw_output_path=None,
            initial_cache_key=None,
            answer_response_hash=_response_hash(responses[-1]),
            answer_raw_output_path=responses[-1].raw_output_path,
            answer_cache_key=responses[-1].cache_key,
            error=error,
        )

    def _request_error_output(
        self, method: str, prompt: str, error: Exception, *, call_count: int = 1
    ) -> MethodOutput:
        return MethodOutput(
            method=method,
            predicted_label="Error",
            raw_model_response="",
            parsed_structure=None,
            validation_errors=(
                {
                    "code": "model_request_error",
                    "message": str(error),
                    "path": "$",
                    "repair_hint": "Inspect the endpoint and saved request audit.",
                    "severity": "hard",
                },
            ),
            repair_triggered=False,
            repair_attempted=False,
            repair_improved=False,
            repair_regression=False,
            repair_regression_layers=(),
            pre_repair_result=None,
            post_repair_result=None,
            proof_trace=(),
            json_parse_valid=None,
            strict_schema_valid=None,
            normalized_schema_valid=None,
            schema_valid=None,
            hard_validator_pass=None,
            soft_issue_count=None,
            solver_executable=None,
            strict_solver_executable=None,
            normalized_solver_executable=None,
            singleton_if_normalized_count=None,
            shadow_solver_label=None,
            answer_from_solver=False,
            route_source="infrastructure_error",
            fallback_used=False,
            coverage=False,
            infrastructure_error=True,
            method_error=False,
            cot_refine_contract_valid=None,
            call_count=call_count,
            latency_ms=0.0,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            cost_usd=0.0,
            model_id=self.client.config.model_id,
            prompt_hash=_hash_prompt(prompt),
            cached=False,
            raw_output_paths=(),
            initial_response_hash=None,
            initial_raw_output_path=None,
            initial_cache_key=None,
            answer_response_hash=None,
            answer_raw_output_path=None,
            answer_cache_key=None,
            error=str(error),
        )


def _hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _safe_input_hash(safe_input: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        safe_input, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _response_hash(response: ModelResponse) -> str:
    return hashlib.sha256(response.content.encode("utf-8")).hexdigest()


def _combine_responses(*responses: ModelResponse) -> ModelResponse:
    if not responses:
        raise ValueError("at least one model response is required")
    last = responses[-1]
    return ModelResponse(
        content=last.content,
        model_id=last.model_id,
        latency_ms=sum(response.latency_ms for response in responses),
        prompt_tokens=sum(response.prompt_tokens for response in responses),
        completion_tokens=sum(response.completion_tokens for response in responses),
        total_tokens=sum(response.total_tokens for response in responses),
        cost_usd=sum(response.cost_usd for response in responses),
        cached=all(response.cached for response in responses),
        raw_output_path=last.raw_output_path,
        provider_payload={
            f"call_{index}": dict(response.provider_payload)
            for index, response in enumerate(responses, start=1)
        },
        raw_output_paths=tuple(
            path for response in responses for path in response.raw_output_paths
        ),
        cache_key=last.cache_key,
    )


def _repair_regression_layers(
    pre: Mapping[str, Any] | None, post: Mapping[str, Any] | None
) -> tuple[str, ...]:
    """Return only explicit True-to-False repair regressions by contract layer."""

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
