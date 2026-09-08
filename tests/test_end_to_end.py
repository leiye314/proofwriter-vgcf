from __future__ import annotations

import json
import unittest
from pathlib import Path

from vgcf.data import Example, SourceSentence
from vgcf.methods import METHODS, MethodRunner
from vgcf.model import ClientConfig, MockModelClient


class EndToEndTests(unittest.TestCase):
    def test_all_methods_run_with_mock(self) -> None:
        root = Path(__file__).resolve().parents[1]
        sample = Example(
            "e2e-1",
            1,
            (
                SourceSentence("s1", "Alice is red."),
                SourceSentence("s2", "If someone is red then they are warm."),
            ),
            SourceSentence("q1", "Alice is warm."),
            "True",
            "test",
        )
        runtime = root / "outputs" / "test_runtime" / "e2e"
        cache_dir = runtime / "cache"
        raw_dir = runtime / "raw"
        cache_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)
        for path in (*cache_dir.glob("*.json"), *raw_dir.glob("*.json")):
            path.unlink()
        client = MockModelClient(
            ClientConfig(
                provider="mock",
                model_id="e2e-mock",
                cache_dir=str(cache_dir),
                raw_dir=str(raw_dir),
            )
        )
        runner = MethodRunner(client, root / "configs" / "prompts")
        outputs = [runner.run(method, sample.model_view()) for method in METHODS]
        self.assertEqual([output.predicted_label for output in outputs], ["True"] * len(METHODS))
        by_method = {output.method: output for output in outputs}
        self.assertTrue(by_method["plain"].parsed_structure)
        self.assertTrue(by_method["constrained"].parsed_structure)
        self.assertTrue(by_method["vgcf2"].proof_trace)
        self.assertTrue(all(output.raw_output_paths for output in outputs))
        constrained_raw = json.loads(
            Path(by_method["constrained"].raw_output_paths[0]).read_text(encoding="utf-8")
        )
        request = constrained_raw["request_payload"]
        self.assertNotIn("response_format", request)
        self.assertEqual(request["temperature"], 0.0)
        visible = request["messages"][1]["content"]
        self.assertIn("s1: Alice is red.", visible)
        self.assertIn("s2: If someone is red then they are warm.", visible)
        self.assertIn("q1: Alice is warm.", visible)


if __name__ == "__main__":
    unittest.main()
