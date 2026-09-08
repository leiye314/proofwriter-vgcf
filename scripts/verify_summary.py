"""Validate the public aggregate; never load predictions or call a model."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SUMMARY_SHA256 = "56e367e89a5cedd5aa473ff004855f4594c6e5e6e21690b19928ace8f259dde2"

def exact_mcnemar(wins: int, losses: int) -> float:
    n = wins + losses
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n)

def verify(value: dict) -> list[str]:
    failures = []
    methods = {row["method"]: row for row in value["methods"]}
    expected = {"direct":114,"cot":236,"cot_refine":245,"constrained":173,
                "gate_cot":263,"repair_only":224,"vgcf2":271}
    if len(value["methods"]) != 7 or set(methods) != set(expected):
        failures.append("seven-method identity")
    for name, correct in expected.items():
        row = methods.get(name, {})
        if row.get("count") != 300 or row.get("correct") != correct:
            failures.append(f"{name}: frozen counts")
        if not math.isclose(row.get("accuracy", -1), correct / 300, abs_tol=1e-14):
            failures.append(f"{name}: accuracy")
    pairs = value["primary_comparisons"]
    if [row["hypothesis_id"] for row in pairs] != ["H1", "H2"]:
        failures.append("hypothesis identity")
    p_values = []
    for row in pairs:
        wins = row["wins_left_correct_right_wrong"]
        losses = row["losses_left_wrong_right_correct"]
        if row["paired_n"] != 300 or wins + losses + row["ties"] != 300:
            failures.append("paired sample accounting")
        if wins - losses != row["left_correct"] - row["right_correct"]:
            failures.append("paired discordance accounting")
        for side in ("left", "right"):
            if row[side+"_correct"] != methods[row[side+"_method"]]["correct"]:
                failures.append("paired method counts")
        if not math.isclose(row["accuracy_delta_left_minus_right"], (wins-losses)/300, abs_tol=1e-14):
            failures.append("paired delta")
        p = exact_mcnemar(wins, losses)
        p_values.append(p)
        if not math.isclose(p, row["mcnemar_exact_p_raw"], abs_tol=1e-14):
            failures.append("exact McNemar")
    previous = 0.0
    for rank, index in enumerate(sorted(range(len(p_values)), key=lambda i:p_values[i])):
        previous = max(previous, min(1.0, (len(p_values)-rank)*p_values[index]))
        if not math.isclose(previous, pairs[index]["mcnemar_exact_p_holm"], abs_tol=1e-14):
            failures.append("Holm adjustment")
    return failures

def verify_file(path: Path) -> list[str]:
    try:
        data = path.read_bytes()
        failures = verify(json.loads(data))
        if hashlib.sha256(data).hexdigest() != EXPECTED_SUMMARY_SHA256:
            failures.append("frozen public aggregate byte identity")
        return failures
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
        return ["summary unavailable or invalid: " + type(exc).__name__]

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=ROOT/"results/frozen_summary.json")
    args = parser.parse_args()
    failures = verify_file(args.summary)
    print(json.dumps({"status":"FAIL" if failures else "PASS", "failures":failures,
                      "scope":"public aggregate arithmetic and byte identity",
                      "bootstrap_intervals":"frozen transcription; not recomputed",
                      "raw_evidence":"unavailable", "model_calls":0}))
    return int(bool(failures))

if __name__ == "__main__":
    raise SystemExit(main())
