from __future__ import annotations

import unittest
from pathlib import Path

from vgcf.metrics import aggregate_records, classification_metrics
from vgcf.analysis import paired_mcnemar
from vgcf.errors import ModelError
from vgcf.model import BaseModelClient, ClientConfig, MockModelClient, ModelResponse


class _FailOnceClient(BaseModelClient):
    def __init__(self, config: ClientConfig) -> None:
        super().__init__(config)
        self.request_count = 0

    def _request(self, payload):
        self.request_count += 1
        if self.request_count == 1:
            raise ModelError("simulated endpoint unavailable")
        return (
            "True",
            {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            {"model": self.config.model_id, "mock": True},
        )


class CacheAndMetricTests(unittest.TestCase):
    def test_infrastructure_failure_is_retried_then_success_is_reused(self) -> None:
        root = (
            Path(__file__).resolve().parents[1]
            / "outputs"
            / "test_runtime"
            / "retry_cache"
        )
        cache_dir = root / "cache"
        raw_dir = root / "raw"
        cache_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)
        for path in (*cache_dir.glob("*.json"), *raw_dir.glob("*.json")):
            path.unlink()
        client = _FailOnceClient(
            ClientConfig(
                provider="offline-test",
                model_id="retry-test",
                cache_dir=str(cache_dir),
                raw_dir=str(raw_dir),
                use_cache=True,
                reasoning_effort="none",
            )
        )
        messages = [
            {"role": "system", "content": "MODE=direct"},
            {"role": "user", "content": "NUMBERED_INPUT\nTHEORY\ns1: Bob is red.\nQUESTION\nq1: Bob is red."},
        ]

        with self.assertRaisesRegex(ModelError, "endpoint unavailable"):
            client.complete(messages, request_tag="failed")
        self.assertEqual(client.request_count, 1)
        self.assertEqual(list(cache_dir.glob("*.json")), [])
        self.assertEqual(list(raw_dir.glob("*.json")), [])

        success = client.complete(messages, request_tag="retried")
        reused = client.complete(messages, request_tag="reused")
        self.assertEqual(client.request_count, 2)
        self.assertFalse(success.cached)
        self.assertTrue(reused.cached)
        self.assertEqual(reused.content, success.content)
        self.assertEqual(reused.cache_key, success.cache_key)

    def test_cache_restores_response_without_second_request(self) -> None:
        root = Path(__file__).resolve().parents[1] / "outputs" / "test_runtime" / "cache_test"
        cache_dir = root / "cache"
        raw_dir = root / "raw"
        cache_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)
        for path in (*cache_dir.glob("*.json"), *raw_dir.glob("*.json")):
            path.unlink()
        client = MockModelClient(
            ClientConfig(
                provider="mock",
                model_id="test-mock",
                cache_dir=str(cache_dir),
                raw_dir=str(raw_dir),
            )
        )
        messages = [
            {"role": "system", "content": "MODE=direct"},
            {
                "role": "user",
                "content": "NUMBERED_INPUT\nTHEORY\ns1: Alice is red.\nQUESTION\nq1: Alice is red.",
            },
        ]
        first = client.complete(messages, request_tag="first")
        second = client.complete(messages, request_tag="second")
        self.assertFalse(first.cached)
        self.assertTrue(second.cached)
        self.assertEqual(first.content, second.content)
        self.assertEqual(second.latency_ms, 0)

    def test_non_thinking_config_reuses_compatible_legacy_cache(self) -> None:
        root = Path(__file__).resolve().parents[1] / "outputs" / "test_runtime" / "legacy_cache"
        cache_dir = root / "cache"
        raw_dir = root / "raw"
        cache_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)
        for path in (*cache_dir.glob("*.json"), *raw_dir.glob("*.json")):
            path.unlink()
        messages = [
            {"role": "system", "content": "MODE=direct"},
            {
                "role": "user",
                "content": "NUMBERED_INPUT\nTHEORY\ns1: Alice is red.\nQUESTION\nq1: Alice is red.",
            },
        ]
        legacy = MockModelClient(
            ClientConfig(
                provider="mock",
                model_id="test-mock",
                cache_dir=str(cache_dir),
                raw_dir=str(raw_dir),
            )
        )
        legacy.complete(messages, request_tag="legacy")
        non_thinking = MockModelClient(
            ClientConfig(
                provider="mock",
                model_id="test-mock",
                cache_dir=str(cache_dir),
                raw_dir=str(raw_dir),
                reasoning_effort="none",
            )
        )
        restored = non_thinking.complete(messages, request_tag="restored")
        self.assertTrue(restored.cached)
        self.assertEqual(restored.content, "True")

    def test_non_thinking_config_rejects_hidden_reasoning_cache(self) -> None:
        root = Path(__file__).resolve().parents[1] / "outputs" / "test_runtime" / "hidden_cache"
        cache_dir = root / "cache"
        raw_dir = root / "raw"
        cache_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)
        for path in (*cache_dir.glob("*.json"), *raw_dir.glob("*.json")):
            path.unlink()
        messages = [
            {"role": "system", "content": "MODE=direct"},
            {
                "role": "user",
                "content": "NUMBERED_INPUT\nTHEORY\ns1: Alice is red.\nQUESTION\nq1: Alice is red.",
            },
        ]
        client = MockModelClient(
            ClientConfig(
                provider="mock",
                model_id="test-mock",
                cache_dir=str(cache_dir),
                raw_dir=str(raw_dir),
                reasoning_effort="none",
            )
        )
        payload = {
            "provider": "mock",
            "model": "test-mock",
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 2000,
            "reasoning_effort": "none",
        }
        key = client.cache.key(payload)
        client.cache.put(
            key,
            ModelResponse(
                content="False",
                model_id="test-mock",
                latency_ms=1,
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
                cost_usd=0,
                cached=False,
                raw_output_path=None,
                provider_payload={
                    "choices": [{"message": {"content": "False", "reasoning_content": "secret"}}]
                },
            ),
        )
        response = client.complete(messages, request_tag="hidden-rejected")
        self.assertFalse(response.cached)
        self.assertEqual(response.content, "True")

    def test_classification_metrics(self) -> None:
        result = classification_metrics(
            ["True", "False", "Unknown", "True"],
            ["True", "Unknown", "Unknown", "False"],
        )
        self.assertEqual(result["accuracy"], 0.5)
        self.assertAlmostEqual(result["per_class"]["Unknown"]["recall"], 1.0)
        self.assertEqual(result["confusion"]["True"]["False"], 1)

    def test_operational_metrics_distinguish_json_and_executability(self) -> None:
        records = [
            {
                "method": "constrained",
                "gold_label": "True",
                "predicted_label": "True",
                "raw_model_response": "{}",
                "parsed_structure": {"facts": [], "rules": [], "query": {}},
                "validation_errors": [],
                "repair_triggered": False,
                "latency_ms": 1,
                "tokens": {"total": 2},
                "cost_usd": 0,
                "cached": False,
                "depth": 0,
                "error": None,
                "json_parse_valid": True,
                "schema_valid": False,
                "hard_validator_pass": False,
                "soft_issue_count": 0,
                "solver_executable": False,
                "repair_attempted": False,
                "repair_improved": False,
                "route_source": "method_error",
                "coverage": True,
                "fallback_used": False,
                "infrastructure_error": False,
                "method_error": False,
                "call_count": 1,
                "final_label_correct": True,
            },
            {
                "method": "constrained",
                "gold_label": "Unknown",
                "predicted_label": "Error",
                "raw_model_response": "not json",
                "parsed_structure": None,
                "validation_errors": [{"code": "invalid_json"}],
                "repair_triggered": False,
                "latency_ms": 1,
                "tokens": {"total": 2},
                "cost_usd": 0,
                "cached": False,
                "depth": 1,
                "error": "invalid json",
                "json_parse_valid": False,
                "schema_valid": False,
                "hard_validator_pass": False,
                "soft_issue_count": 0,
                "solver_executable": False,
                "repair_attempted": False,
                "repair_improved": False,
                "route_source": "method_error",
                "coverage": False,
                "fallback_used": False,
                "infrastructure_error": False,
                "method_error": True,
                "call_count": 1,
                "final_label_correct": False,
            },
        ]
        metrics = aggregate_records(records)["constrained"]
        self.assertEqual(metrics["json_parse_valid_rate"], 0.5)
        self.assertEqual(metrics["schema_valid_rate"], 0.0)
        self.assertEqual(metrics["solver_executable_rate"], 0.0)
        self.assertEqual(metrics["validation_issue_count"], 1)

    def test_route_accuracy_false_blocks_and_wilson_intervals(self) -> None:
        rows = []
        for route, predicted, answer_from_solver in (
            ("solver_initial", "True", True),
            ("solver_repair", "False", True),
            ("cot_fallback", "True", False),
        ):
            rows.append(
                {
                    "method": "gate_cot",
                    "gold_label": "True",
                    "predicted_label": predicted,
                    "final_label_correct": predicted == "True",
                    "route_source": route,
                    "answer_from_solver": answer_from_solver,
                    "coverage": True,
                    "fallback_used": route == "cot_fallback",
                    "pre_repair_result": {
                        "solver_executable": True,
                        "solver_label": "True",
                    },
                    "validation_errors": [],
                    "repair_attempted": False,
                    "repair_improved": False,
                    "json_parse_valid": True,
                    "schema_valid": True,
                    "hard_validator_pass": route != "cot_fallback",
                    "soft_issue_count": 0,
                    "solver_executable": True,
                    "shadow_solver_label": "True",
                    "infrastructure_error": False,
                    "method_error": False,
                    "call_count": 1,
                    "latency_ms": 1,
                    "tokens": {"prompt": 1, "completion": 1, "total": 2},
                    "cost_usd": 0,
                    "cached": False,
                    "depth": 1,
                }
            )
        metrics = aggregate_records(rows)["gate_cot"]
        self.assertEqual(metrics["solver_initial_count"], 1)
        self.assertEqual(metrics["solver_repair_count"], 1)
        self.assertEqual(metrics["cot_fallback_count"], 1)
        self.assertEqual(metrics["accuracy_by_route_source"]["cot_fallback"]["accuracy"], 1.0)
        self.assertEqual(metrics["gate_false_block_rate"], 1.0)
        self.assertLess(metrics["accuracy_wilson_95"]["low"], metrics["accuracy"])

    def test_paired_mcnemar_reports_discordant_inputs(self) -> None:
        rows = [
            {"example_id": "e1", "method": "a", "gold_label": "True", "predicted_label": "True"},
            {"example_id": "e1", "method": "b", "gold_label": "True", "predicted_label": "False"},
            {"example_id": "e2", "method": "a", "gold_label": "False", "predicted_label": "True"},
            {"example_id": "e2", "method": "b", "gold_label": "False", "predicted_label": "False"},
        ]
        result = paired_mcnemar(rows)["a__vs__b"]
        self.assertEqual(result["a_correct_b_wrong"], 1)
        self.assertEqual(result["a_wrong_b_correct"], 1)
        self.assertEqual(result["discordant_count"], 2)
        self.assertEqual(result["exact_two_sided_p"], 1.0)


if __name__ == "__main__":
    unittest.main()
