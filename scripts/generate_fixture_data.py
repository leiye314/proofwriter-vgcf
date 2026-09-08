"""Generate the deterministic controlled-language smoke dataset."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vgcf.controlled_language import formalize_controlled  # noqa: E402
from vgcf.solver import ForwardChainingSolver  # noqa: E402


def build_records() -> list[dict[str, object]]:
    names = ["Alice", "Bob", "Cara", "Dan", "Erin", "Finn"]
    others = ["Bob", "Cara", "Dan", "Erin", "Finn", "Alice"]
    colors = ["red", "blue", "green", "golden", "calm", "furry"]
    outcomes = ["warm", "cool", "bright", "kind", "quiet", "soft"]
    records: list[dict[str, object]] = []
    cases: list[tuple[list[str], str, int]] = []
    for name, other, color, outcome in zip(names, others, colors, outcomes):
        cases.extend(
            [
                (
                    [
                        f"{name} is {color}.",
                        f"If someone is {color} then they are {outcome}.",
                    ],
                    f"{name} is {outcome}.",
                    1,
                ),
                ([f"{name} is not {outcome}."], f"{name} is {outcome}.", 0),
                ([f"{name} is {color}."], f"{name} is {outcome}.", 0),
                (
                    [
                        f"{name} is {color}.",
                        f"{name} is young.",
                        f"If someone is {color} and they are young then they are kind.",
                    ],
                    f"{name} is kind.",
                    1,
                ),
                ([f"{name} likes {other}."], f"{other} likes {name}.", 0),
                (
                    [
                        f"{name} is {color}.",
                        f"If someone is {color} then they are warm.",
                        "If someone is warm then they are kind.",
                    ],
                    f"{name} is kind.",
                    2,
                ),
            ]
        )
    solver = ForwardChainingSolver()
    for index, (theory, question, depth) in enumerate(cases, start=1):
        gold = solver.solve(formalize_controlled(theory, question)).label
        records.append(
            {
                "example_id": f"fixture-{index:03d}",
                "depth": depth,
                "theory": theory,
                "question": question,
                "gold_label": gold,
            }
        )
    return records


def main() -> int:
    output = ROOT / "data" / "fixtures" / "proofwriter_smoke.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    records = build_records()
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "records": len(records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
