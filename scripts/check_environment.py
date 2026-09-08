"""Preflight for the independent public offline workflow."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-final-evidence", action="store_true")
    args = parser.parse_args()
    failures = []
    if sys.version_info < (3, 12):
        failures.append("Python 3.12 or newer is required")
    try:
        manifest = json.loads((ROOT/"PUBLIC_EXPORT_MANIFEST.json").read_text(encoding="utf-8"))
        for entry in manifest["copied_unchanged"]:
            path = ROOT/entry["path"]
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                failures.append("missing or changed export file: " + entry["path"])
    except (OSError, ValueError, KeyError):
        failures.append("export manifest unavailable or malformed")
    if args.require_final_evidence:
        failures.append("Final evidence is not distributed in the public repository")
    print(json.dumps({"status":"FAIL" if failures else "PASS", "failures":failures,
                      "final_evidence":"unavailable", "external_model_calls":0}))
    return int(bool(failures))

if __name__ == "__main__":
    raise SystemExit(main())
