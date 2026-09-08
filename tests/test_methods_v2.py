from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any, Mapping

from vgcf.data import Example, SourceSentence, load_examples
from vgcf.ir_v2 import parse_v2_ir
from vgcf.methods import MethodRunner
from vgcf.errors import ModelError
from vgcf.model import BaseModelClient, ClientConfig, MockModelClient


class CountingMockModelClient(MockModelClient):
    def __init__(self, config: ClientConfig) -> None:
        super().__init__(config)
        self.request_count = 0

    def _request(
        self, payload: Mapping[str, Any]
    ) -> tuple[str, Mapping[str, Any], Mapping[str, Any]]:
        self.request_count += 1
        return super()._request(payload)


class FailedRepairClient(BaseModelClient):
    def __init__(self, config: ClientConfig) -> None:
        super().__init__(config)
        self.payloads: list[Mapping[str, Any]] = []

    def _request(
        self, payload: Mapping[str, Any]
    ) -> tuple[str, Mapping[str, Any], Mapping[str, Any]]:
        self.payloads.append(payload)
        system = "\n".join(
            item["content"] for item in payload["messages"] if item["role"] == "system"
        )
        if "REPAIR_REQUEST" in system:
            content = "not json"
        elif "MODE=cot" in system:
            content = (
                "REASONING:\nChecked forward only.\n\n"
                "FINAL_LABEL_FOR_ORIGINAL_QUESTION: True"
            )
        else:
            content = json.dumps(
                {
                    "facts": [{"id": "s999", "atom": "+red(alice)"}],
                    "rules": [],
                    "query": {"id": "q1", "atom": "+red(alice)"},
                }
            )
        return content, {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}, {"mock": True}


class RepairExceptionClient(FailedRepairClient):
    def _request(
        self, payload: Mapping[str, Any]
    ) -> tuple[str, Mapping[str, Any], Mapping[str, Any]]:
        system = "\n".join(
            item["content"] for item in payload["messages"] if item["role"] == "system"
        )
        if "REPAIR_REQUEST" in system:
            self.payloads.append(payload)
            raise ModelError("synthetic repair outage")
        return super()._request(payload)


class SuccessfulRepairClient(FailedRepairClient):
    def _request(
        self, payload: Mapping[str, Any]
    ) -> tuple[str, Mapping[str, Any], Mapping[str, Any]]:
        system = "\n".join(
            item["content"] for item in payload["messages"] if item["role"] == "system"
        )
        if "REPAIR_REQUEST" not in system:
            return super()._request(payload)
        self.payloads.append(payload)
        content = json.dumps(
            {
                "facts": [{"id": "s1", "atom": "+red(alice)"}],
                "rules": [
                    {
                        "id": "s2",
                        "if": ["+red(X)"],
                        "then": "+warm(X)",
                    }
                ],
                "query": {"id": "q1", "atom": "+warm(alice)"},
            }
        )
        return content, {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}, {"mock": True}


class ExecutableHardInvalidRepairClient(FailedRepairClient):
    """Initial is unusable; repair executes but still omits one source sentence."""

    def _request(
        self, payload: Mapping[str, Any]
    ) -> tuple[str, Mapping[str, Any], Mapping[str, Any]]:
        self.payloads.append(payload)
        system = "\n".join(
            item["content"] for item in payload["messages"] if item["role"] == "system"
        )
        if "MODE=cot" in system:
            content = (
                "REASONING:\nChecked forward only.\n\n"
                "FINAL_LABEL_FOR_ORIGINAL_QUESTION: True"
            )
        elif "REPAIR_REQUEST" in system:
            content = json.dumps(
                {
                    "facts": [{"id": "s1", "atom": "+red(alice)"}],
                    "rules": [],
                    "query": {"id": "q1", "atom": "+warm(alice)"},
                }
            )
        else:
            content = json.dumps(
                {
                    "facts": [{"id": "s1", "atom": "+red(X)"}],
                    "rules": [],
                    "query": {"id": "q1", "atom": "+warm(alice)"},
                }
            )
        return content, {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}, {"mock": True}


class MethodsV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[1]
        self.runtime = self.root / "outputs" / "test_runtime" / "methods_v2"
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.sample = Example(
            "sample:Q1",
            1,
            (
                SourceSentence("s1", "Alice is red."),
                SourceSentence("s2", "If someone is red then they are warm."),
            ),
            SourceSentence("q1", "Alice is warm."),
            "True",
            "test",
            "sample",
            "meta-dev",
        )

    def _config(self, name: str, *, use_cache: bool = False) -> ClientConfig:
        return ClientConfig(
            provider="mock",
            model_id="v2-test",
            cache_dir=str(self.runtime / name / "cache"),
            raw_dir=str(self.runtime / name / "raw"),
            use_cache=use_cache,
        )

    def test_numbered_lines_are_in_final_visible_request(self) -> None:
        client = MockModelClient(self._config("numbered"))
        runner = MethodRunner(client, self.root / "configs" / "prompts", phase="dev")
        output = runner.run("constrained", self.sample.model_view())
        raw = json.loads(Path(output.raw_output_paths[0]).read_text(encoding="utf-8"))
        visible = raw["request_payload"]["messages"][1]["content"]
        self.assertIn("s1: Alice is red.", visible)
        self.assertIn("s2: If someone is red then they are warm.", visible)
        self.assertIn("q1: Alice is warm.", visible)

    def test_plain_and_constrained_declare_same_ir_contract(self) -> None:
        plain = (self.root / "configs" / "prompts" / "plain.txt").read_text(encoding="utf-8")
        constrained = (self.root / "configs" / "prompts" / "constrained.txt").read_text(encoding="utf-8")
        self.assertIn("IR_CONTRACT=vgcf-ir-v2", plain)
        self.assertIn("IR_CONTRACT=vgcf-ir-v2", constrained)
        self.assertNotIn("Project-authored toy example", plain)
        self.assertIn("Project-authored toy example", constrained)
        client = MockModelClient(self._config("same_contract"))
        runner = MethodRunner(client, self.root / "configs" / "prompts", phase="dev")
        plain_output = runner.run("plain", self.sample.model_view())
        constrained_output = runner.run("constrained", self.sample.model_view())
        self.assertEqual(
            parse_v2_ir(plain_output.raw_model_response).to_dict(),
            parse_v2_ir(constrained_output.raw_model_response).to_dict(),
        )

    def test_vgcf2_families_reuse_identical_initial_response(self) -> None:
        client = CountingMockModelClient(self._config("reuse"))
        runner = MethodRunner(client, self.root / "configs" / "prompts", phase="dev")
        methods = ("constrained", "gate_cot", "repair_only", "vgcf2")
        outputs = [runner.run(method, self.sample.model_view()) for method in methods]
        self.assertEqual(client.request_count, 1)
        self.assertEqual(len({output.initial_response_hash for output in outputs}), 1)
        self.assertEqual(len({output.raw_output_paths[0] for output in outputs}), 1)

    def test_failed_repair_cannot_overwrite_executable_initial(self) -> None:
        client = FailedRepairClient(self._config("failed_repair"))
        runner = MethodRunner(client, self.root / "configs" / "prompts", phase="dev")
        output = runner.run("repair_only", self.sample.model_view())
        self.assertEqual(output.predicted_label, "True")
        self.assertEqual(output.route_source, "solver_initial")
        self.assertTrue(output.solver_executable)
        self.assertFalse(output.repair_improved)
        self.assertIsNotNone(output.post_repair_result)
        assert output.parsed_structure is not None
        self.assertEqual(output.parsed_structure["facts"][0]["source_id"], "s999")

    def test_plain_and_constrained_report_hard_invalid_executable_solver_label(self) -> None:
        client = FailedRepairClient(self._config("ungated_baselines"))
        runner = MethodRunner(client, self.root / "configs" / "prompts", phase="dev")
        for method in ("plain", "constrained"):
            output = runner.run(method, self.sample.model_view())
            self.assertEqual(output.predicted_label, "True")
            self.assertEqual(output.route_source, "solver_initial")
            self.assertFalse(output.hard_validator_pass)
            self.assertTrue(output.solver_executable)
            self.assertTrue(output.answer_from_solver)

    def test_gate_fallback_reuses_exact_independent_cot_response(self) -> None:
        client = FailedRepairClient(self._config("gate_cot_reuse"))
        runner = MethodRunner(client, self.root / "configs" / "prompts", phase="dev")
        cot = runner.run("cot", self.sample.model_view())
        gated = runner.run("gate_cot", self.sample.model_view())
        self.assertEqual(gated.route_source, "cot_fallback")
        self.assertTrue(gated.solver_executable)
        self.assertFalse(gated.answer_from_solver)
        self.assertEqual(gated.answer_response_hash, cot.answer_response_hash)
        self.assertEqual(gated.answer_raw_output_path, cot.answer_raw_output_path)
        self.assertEqual(gated.answer_cache_key, cot.answer_cache_key)

    def test_repair_only_has_no_cot_fallback_and_separates_advisories(self) -> None:
        client = FailedRepairClient(self._config("repair_feedback"))
        runner = MethodRunner(client, self.root / "configs" / "prompts", phase="dev")
        output = runner.run("repair_only", self.sample.model_view())
        self.assertEqual(output.call_count, 2)
        systems = [
            "\n".join(
                item["content"]
                for item in payload["messages"]
                if item["role"] == "system"
            )
            for payload in client.payloads
        ]
        self.assertFalse(any("MODE=cot" in system for system in systems))
        repair_user = client.payloads[-1]["messages"][-1]["content"]
        self.assertIn("HARD_ISSUES_MANDATORY", repair_user)
        self.assertIn("SOFT_WARNINGS_ADVISORY_NON_MANDATORY", repair_user)

    def test_repair_exception_counts_attempt_and_still_reuses_cot(self) -> None:
        client = RepairExceptionClient(self._config("repair_exception"))
        runner = MethodRunner(client, self.root / "configs" / "prompts", phase="dev")
        cot = runner.run("cot", self.sample.model_view())
        output = runner.run("vgcf2", self.sample.model_view())
        self.assertTrue(output.repair_attempted)
        self.assertEqual(output.call_count, 3)
        self.assertEqual(output.route_source, "cot_fallback")
        self.assertTrue(output.infrastructure_error)
        self.assertEqual(output.answer_response_hash, cot.answer_response_hash)
        self.assertEqual(output.answer_raw_output_path, cot.answer_raw_output_path)
        self.assertEqual(output.answer_cache_key, cot.answer_cache_key)

    def test_repair_only_best_candidate_drives_the_output(self) -> None:
        client = SuccessfulRepairClient(self._config("successful_repair"))
        runner = MethodRunner(client, self.root / "configs" / "prompts", phase="dev")
        output = runner.run("repair_only", self.sample.model_view())
        self.assertEqual(output.route_source, "solver_repair")
        self.assertTrue(output.repair_improved)
        self.assertTrue(output.hard_validator_pass)
        assert output.parsed_structure is not None
        self.assertEqual(output.parsed_structure["facts"][0]["source_id"], "s1")
        self.assertEqual(output.parsed_structure["rules"][0]["source_id"], "s2")

    def test_repair_only_can_select_executable_hard_invalid_repair(self) -> None:
        client = ExecutableHardInvalidRepairClient(
            self._config("executable_hard_invalid_repair_only")
        )
        runner = MethodRunner(client, self.root / "configs" / "prompts", phase="dev")

        output = runner.run("repair_only", self.sample.model_view())

        self.assertEqual(output.route_source, "solver_repair")
        self.assertEqual(output.predicted_label, "Unknown")
        self.assertTrue(output.repair_attempted)
        self.assertTrue(output.repair_improved)
        self.assertTrue(output.json_parse_valid)
        self.assertTrue(output.schema_valid)
        self.assertFalse(output.hard_validator_pass)
        self.assertTrue(output.solver_executable)
        assert output.post_repair_result is not None
        self.assertFalse(output.post_repair_result["hard_validator_pass"])
        self.assertTrue(output.post_repair_result["solver_executable"])

    def test_full_vgcf2_falls_back_for_executable_hard_invalid_repair(self) -> None:
        client = ExecutableHardInvalidRepairClient(
            self._config("executable_hard_invalid_full")
        )
        runner = MethodRunner(client, self.root / "configs" / "prompts", phase="dev")

        output = runner.run("vgcf2", self.sample.model_view())

        self.assertEqual(output.route_source, "cot_fallback")
        self.assertEqual(output.predicted_label, "True")
        self.assertTrue(output.repair_attempted)
        self.assertTrue(output.fallback_used)
        assert output.post_repair_result is not None
        self.assertFalse(output.post_repair_result["hard_validator_pass"])
        self.assertTrue(output.post_repair_result["solver_executable"])

    def test_full_vgcf2_routes_invalid_repair_to_same_model_cot(self) -> None:
        client = FailedRepairClient(self._config("fallback"))
        runner = MethodRunner(client, self.root / "configs" / "prompts", phase="dev")
        output = runner.run("vgcf2", self.sample.model_view())
        self.assertEqual(output.predicted_label, "True")
        self.assertEqual(output.route_source, "cot_fallback")
        self.assertTrue(output.repair_attempted)
        self.assertTrue(output.fallback_used)
        self.assertEqual(output.call_count, 3)

    def test_gold_sentinel_never_reaches_raw_or_cache_requests(self) -> None:
        sentinel = "GOLD_SENTINEL_70AF2E"
        record = {
            "id": "sentinel-story",
            "triples": {
                "triple1": {"text": "Alice is red.", "representation": sentinel}
            },
            "rules": {},
            "questions": {
                "Q1": {
                    "question": "Alice is red.",
                    "answer": True,
                    "representation": sentinel,
                    "proofs": sentinel,
                }
            },
        }
        path = self.runtime / "sentinel.jsonl"
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        config = self._config("sentinel", use_cache=True)
        for directory in (Path(config.cache_dir), Path(config.raw_dir)):
            directory.mkdir(parents=True, exist_ok=True)
            for old in directory.glob("*.json"):
                old.unlink()
        client = MockModelClient(config)
        runner = MethodRunner(client, self.root / "configs" / "prompts")
        runner.run("constrained", load_examples(path)[0].model_view())
        audited = list(Path(config.cache_dir).glob("*.json")) + list(
            Path(config.raw_dir).glob("*.json")
        )
        self.assertTrue(audited)
        self.assertTrue(
            all(sentinel not in item.read_text(encoding="utf-8") for item in audited)
        )


if __name__ == "__main__":
    unittest.main()
