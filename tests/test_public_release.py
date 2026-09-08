from __future__ import annotations
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from scripts.verify_summary import ROOT, exact_mcnemar, verify, verify_file

class PublicReleaseTests(unittest.TestCase):
    def setUp(self):
        self.value = json.loads((ROOT/"results/frozen_summary.json").read_text())

    def test_frozen_aggregate(self):
        self.assertEqual(verify_file(ROOT/"results/frozen_summary.json"), [])

    def test_corrupted_count(self):
        self.value["methods"][0]["correct"] += 1
        self.assertTrue(verify(self.value))

    def test_corrupted_paired_accounting(self):
        self.value["primary_comparisons"][0]["ties"] -= 1
        self.assertIn("paired sample accounting", verify(self.value))

    def test_corrupted_adjustment(self):
        self.value["primary_comparisons"][0]["mcnemar_exact_p_holm"] = 0.9
        self.assertIn("Holm adjustment", verify(self.value))

    def test_missing_and_malformed_summary_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"summary.json"
            self.assertTrue(verify_file(path))
            path.write_text("{}")
            self.assertTrue(verify_file(path))

    def test_zero_discordance(self):
        self.assertEqual(exact_mcnemar(0, 0), 1.0)

    def test_private_evidence_is_explicitly_unavailable(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT/"scripts/check_environment.py"),
                                 "--require-final-evidence"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stdout)["final_evidence"], "unavailable")
