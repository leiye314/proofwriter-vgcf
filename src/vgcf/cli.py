"""Command-line entry points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import analyze_results
from .experiment import load_config, run_experiment


def run_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a VGCF experiment")
    parser.add_argument("--config", default="configs/smoke.json")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args(argv)
    config, config_hash = load_config(args.config)
    records = run_experiment(config, config_hash, Path(args.project_root))
    print(json.dumps({"status": "ok", "records": len(records), "output": config.output_path}))
    return 0


def analyze_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze cached VGCF results")
    parser.add_argument("--input", default="outputs/smoke_results.jsonl")
    parser.add_argument("--output-dir", default="outputs/analysis")
    args = parser.parse_args(argv)
    summary = analyze_results(args.input, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0
