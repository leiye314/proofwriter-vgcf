from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vgcf.data import inspect_jsonl_schema, inspect_proofwriter_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect JSONL field names and types before choosing an adapter"
    )
    parser.add_argument("path")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument(
        "--full-proofwriter",
        action="store_true",
        help="inspect the complete documented nested ProofWriter split",
    )
    parser.add_argument("--json-output")
    parser.add_argument("--markdown-output")
    args = parser.parse_args()
    report = (
        inspect_proofwriter_dataset(args.path)
        if args.full_proofwriter
        else inspect_jsonl_schema(args.path, args.limit)
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True)
    print(rendered)
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    if args.markdown_output:
        if not args.full_proofwriter:
            parser.error("--markdown-output requires --full-proofwriter")
        output = Path(args.markdown_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(_render_markdown(report), encoding="utf-8")
    return 0


def _render_markdown(report: dict[str, object]) -> str:
    root_fields = ", ".join(f"`{key}`" for key in report["root_fields"])
    question_fields = ", ".join(
        f"`{value}`" for value in report["question_item_fields"]
    )
    labels = ", ".join(
        f"{key}: {value}" for key, value in report["label_distribution"].items()
    )
    depths = ", ".join(
        f"{key}: {value}" for key, value in report["depth_distribution"].items()
    )
    theory_examples = "\n".join(
        f"- `{value}`" for value in report["theory_examples"]
    )
    question_examples = "\n".join(
        f"- `{value}`" for value in report["question_examples"]
    )
    return f"""# Real ProofWriter data inspection

This inspection is generated from the complete local split, not from inferred field names.

## Source and identity

- Local path: `{report['path']}`
- SHA-256: `{report['sha256']}`
- Detected profile: `{', '.join(report['source_profile'])}`
- World assumption: `{report['world_assumption']}`
- Theories: {report['record_count']}
- Questions: {report['question_count']}

The authoritative release is AI2 ProofWriter V2020.12.3. The experiment uses the OWA depth-5 test split. Gold annotations remain evaluator-only.

- Official landing page: https://allenai.org/data/proofwriter
- Versioned release URL: https://aristo-data-public.s3.amazonaws.com/proofwriter/proofwriter-dataset-V2020.12.3.zip
- Release archive size: 214,185,889 bytes
- Release archive SHA-256: `bbc5694901e8306d0bd659aa1ad53ccfd02c201864f4b320ffa3777827d1fc26`
- Evaluated split: `OWA/depth-5/meta-test.jsonl`
- Evaluated split SHA-256: `{report['sha256']}`

## Actual fields and formats

- Root fields: {root_fields}
- Triple item fields: {', '.join(f'`{value}`' for value in report['triple_item_fields'])}
- Rule item fields: {', '.join(f'`{value}`' for value in report['rule_item_fields'])}
- Question item fields: {question_fields}
- Theory: {report['theory_format']}
- Question: {report['question_format']}

Theory excerpts:

{theory_examples}

Question excerpts:

{question_examples}

## Labels, depth, and semantics

- Label distribution: {labels}
- QDep distribution: {depths}
- OWA/CWA consistency check: {report['assumption_consistent']}
- Theory string vs. ordered triple/rule text mismatches: {report['theory_text_mismatch_count']}
- Declared NFact/NRule count mismatches: {report['declared_count_mismatch_count']}
- Unexpected root fields: {report['unexpected_root_fields']}

Under OWA, an unproved query is `Unknown`; it is not converted to `False`. Explicit negative facts and conclusions are signed atoms, not negation-as-failure.

## Leakage boundary

The only model-visible keys are `{report['model_input_whitelist']}`. The nested adapter reads natural-language `text` and `question`; `answer`, proofs, `allProofs`, formal `representation`, proof details, strategies, and arbitrary metadata never enter a model request. Labels and depth are used only for evaluator-side sampling and scoring.
"""


if __name__ == "__main__":
    raise SystemExit(main())
