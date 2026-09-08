"""Run a deterministic, network-free mock smoke entirely in a temporary directory."""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import io
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vgcf.analysis import analyze_results  # noqa: E402
from vgcf.experiment import load_config, run_experiment  # noqa: E402


PROJECTION_FIELDS = (
    "example_id",
    "depth",
    "gold_label",
    "predicted_label",
    "method",
    "raw_model_response",
    "parsed_structure",
    "validation_errors",
    "repair_triggered",
    "repair_attempted",
    "repair_improved",
    "repair_regression",
    "pre_repair_result",
    "post_repair_result",
    "proof_trace",
    "json_parse_valid",
    "strict_schema_valid",
    "normalized_schema_valid",
    "hard_validator_pass",
    "solver_executable",
    "strict_solver_executable",
    "normalized_solver_executable",
    "shadow_solver_label",
    "answer_from_solver",
    "final_label_correct",
    "route_source",
    "coverage",
    "fallback_used",
    "method_error",
    "model_id",
    "prompt_hash",
    "source_profile",
    "model_input",
    "model_input_hash",
    "phase",
    "source_split",
    "theory_id",
    "dataset_sha256",
    "sample_set",
)
EXPECTED_RECORD_COUNT = 150
EXPECTED_EXAMPLE_COUNT = 30
EXPECTED_METHOD_COUNT = 5
EXPECTED_PROGRESS_EVENT_COUNT = 30
EXPECTED_PROJECTION_SHA256 = (
    "13814cb47bec367fe28ee776e9212e25f5828e5fbd0c36b69a641ece225b55b9"
)
EXPECTED_ANALYSIS_SHA256 = (
    "ff514edd921acf84ee13141d7ece55e0f0b870fe20193947bba836424d8ab35c"
)


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _projection(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: record.get(key) for key in PROJECTION_FIELDS}
        for record in sorted(records, key=lambda row: (row["example_id"], row["method"]))
    ]


def _contract_failures(run: dict[str, Any]) -> list[str]:
    expected = {
        "record_count": EXPECTED_RECORD_COUNT,
        "example_count": EXPECTED_EXAMPLE_COUNT,
        "method_count": EXPECTED_METHOD_COUNT,
        "progress_event_count": EXPECTED_PROGRESS_EVENT_COUNT,
        "projection_sha256": EXPECTED_PROJECTION_SHA256,
        "analysis_method_metrics_sha256": EXPECTED_ANALYSIS_SHA256,
        "method_error_count": 0,
        "error_prediction_count": 0,
    }
    return [
        f"{key}: expected {expected_value!r}, got {run.get(key)!r}"
        for key, expected_value in expected.items()
        if run.get(key) != expected_value
    ]


def _run_once(root: Path, smoke: dict[str, Any], run_name: str) -> dict[str, Any]:
    run_root = root / run_name
    run_root.mkdir()
    config_value = copy.deepcopy(smoke)
    config_value["dataset_path"] = str(
        (PROJECT_ROOT / "data/fixtures/proofwriter_smoke.jsonl").resolve()
    )
    config_value["prompt_directory"] = str((PROJECT_ROOT / "configs/prompts").resolve())
    config_value["output_path"] = str((run_root / "results.jsonl").resolve())
    config_value["client"]["cache_dir"] = str((run_root / "cache").resolve())
    config_value["client"]["raw_dir"] = str((run_root / "raw").resolve())
    # Each run has its own empty cache; keeping cache persistence enabled lets
    # the offline physical-accounting path verify raw/cache cross-bindings.
    config_value["client"]["use_cache"] = True
    if config_value["client"]["provider"] != "mock":
        raise ValueError("fixture smoke refuses every provider except mock")
    config_path = run_root / "config.json"
    config_path.write_text(
        json.dumps(config_value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    config, config_hash = load_config(config_path)
    progress = io.StringIO()
    with contextlib.redirect_stderr(progress):
        records = run_experiment(config, config_hash, PROJECT_ROOT)
    analysis = analyze_results(run_root / "results.jsonl", run_root / "analysis")
    deterministic_analysis = {
        method: {
            key: value
            for key, value in metrics.items()
            if "latency" not in key
        }
        for method, metrics in analysis["methods"].items()
    }
    projected = _projection(records)
    return {
        "record_count": len(records),
        "example_count": len({item["example_id"] for item in records}),
        "method_count": len({item["method"] for item in records}),
        "projection": projected,
        "projection_sha256": _canonical_hash(projected),
        "analysis_metrics": deterministic_analysis,
        "analysis_method_metrics_sha256": _canonical_hash(deterministic_analysis),
        "progress_event_count": len(progress.getvalue().splitlines()),
        "method_error_count": sum(bool(item.get("method_error")) for item in records),
        "error_prediction_count": sum(
            item.get("predicted_label") == "Error" for item in records
        ),
        "result_exists": (run_root / "results.jsonl").is_file(),
        "manifest_exists": (run_root / "results.manifest.json").is_file(),
        "analysis_exists": (run_root / "analysis/summary.json").is_file(),
    }


def run_smoke(project_root: Path) -> dict[str, Any]:
    resolved = project_root.resolve()
    if resolved != PROJECT_ROOT.resolve():
        raise ValueError(
            f"this script belongs to {PROJECT_ROOT.resolve()}, not {resolved}"
        )
    smoke = json.loads((resolved / "configs/smoke.json").read_text(encoding="utf-8"))
    if smoke.get("client", {}).get("provider") != "mock":
        raise ValueError("configs/smoke.json must use the deterministic mock provider")
    temporary_path: Path | None = None
    with tempfile.TemporaryDirectory(prefix="vgcf_portfolio_smoke_") as directory:
        temporary_path = Path(directory)
        first = _run_once(temporary_path, smoke, "run_a")
        second = _run_once(temporary_path, smoke, "run_b")
        deterministic = (
            first["projection_sha256"] == second["projection_sha256"]
            and first["analysis_method_metrics_sha256"]
            == second["analysis_method_metrics_sha256"]
        )
        artifacts_complete = all(
            first[key] and second[key]
            for key in ("result_exists", "manifest_exists", "analysis_exists")
        )
        contract_failures = {
            "run_a": _contract_failures(first),
            "run_b": _contract_failures(second),
        }
        contract_matches = not any(contract_failures.values())
        report = {
            "status": (
                "PASS"
                if deterministic and artifacts_complete and contract_matches
                else "FAIL"
            ),
            "operation": "synthetic_fixture_software_smoke",
            "classification": (
                "software QA on project-generated synthetic data; not ProofWriter "
                "benchmark evidence and not an LLM experiment"
            ),
            "project_root": str(resolved),
            "runs": {
                "run_a": {
                    key: value
                    for key, value in first.items()
                    if key not in {"projection", "analysis_metrics"}
                },
                "run_b": {
                    key: value
                    for key, value in second.items()
                    if key not in {"projection", "analysis_metrics"}
                },
            },
            "analysis_metric_differences": (
                {
                    method: {
                        "run_a": first["analysis_metrics"][method],
                        "run_b": second["analysis_metrics"][method],
                    }
                    for method in first["analysis_metrics"]
                    if first["analysis_metrics"][method]
                    != second["analysis_metrics"][method]
                }
                if first["analysis_method_metrics_sha256"]
                != second["analysis_method_metrics_sha256"]
                else {}
            ),
            "deterministic_semantic_projection": deterministic,
            "expected_fixture_contract_matches": contract_matches,
            "contract_failures": contract_failures,
            "artifacts_complete_before_cleanup": artifacts_complete,
            "external_model_calls_made": 0,
            "network_calls_made": 0,
            "frozen_final_outputs_read": 0,
            "frozen_final_outputs_written": 0,
            "repository_files_written": 0,
            "temporary_directory": str(temporary_path),
        }
    assert temporary_path is not None
    report["temporary_directory_cleaned"] = not temporary_path.exists()
    if not report["temporary_directory_cleaned"]:
        report["status"] = "FAIL"
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run network-free VGCF fixture smoke")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args(argv)
    try:
        report = run_smoke(args.project_root)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        report = {
            "status": "FAIL",
            "operation": "synthetic_fixture_software_smoke",
            "error": str(exc),
            "external_model_calls_made": 0,
            "network_calls_made": 0,
            "repository_files_written": 0,
        }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
