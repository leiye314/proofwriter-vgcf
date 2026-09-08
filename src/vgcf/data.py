"""ProofWriter data adapters, inspection, leakage guard, and seeded sampling."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .errors import DatasetSchemaError

MODEL_INPUT_FIELDS = frozenset({"theory", "question"})
FORBIDDEN_DATASET_FIELDS = frozenset(
    {
        "answer",
        "label",
        "gold_label",
        "allProofs",
        "proof",
        "proofs",
        "proofsWithIntermediates",
        "representation",
        "hidden_rules",
        "metadata",
    }
)


@dataclass(frozen=True, slots=True)
class SourceSentence:
    source_id: str
    text: str


@dataclass(frozen=True, slots=True)
class Example:
    example_id: str
    depth: int | None
    theory: tuple[SourceSentence, ...]
    question: SourceSentence
    gold_label: str
    source_profile: str
    theory_id: str = ""
    source_split: str = "unknown"

    def model_view(self) -> "ModelExample":
        """Drop every evaluator-only value before method execution."""

        return ModelExample(
            example_id=self.example_id,
            theory=self.theory,
            question=self.question,
            source_profile=self.source_profile,
            theory_id=self.theory_id,
            source_split=self.source_split,
        )


@dataclass(frozen=True, slots=True)
class ModelExample:
    """The complete type accepted by methods; it contains no gold fields."""

    example_id: str
    theory: tuple[SourceSentence, ...]
    question: SourceSentence
    source_profile: str
    theory_id: str = ""
    source_split: str = "unknown"


def build_model_input(example: Example | ModelExample) -> dict[str, Any]:
    """The only gateway from a dataset example to model-visible content."""

    payload = {
        "theory": [sentence.text for sentence in example.theory],
        "question": example.question.text,
    }
    assert_safe_model_input(payload)
    return payload


def assert_safe_model_input(payload: Mapping[str, Any]) -> None:
    keys = set(payload)
    if keys != MODEL_INPUT_FIELDS:
        raise DatasetSchemaError(
            "model input must contain exactly theory and question; "
            f"received {sorted(keys)}"
        )
    if not isinstance(payload["theory"], list) or not all(
        isinstance(item, str) for item in payload["theory"]
    ):
        raise DatasetSchemaError("model input theory must be a list of strings")
    if not isinstance(payload["question"], str):
        raise DatasetSchemaError("model input question must be a string")


def render_model_input(payload: Mapping[str, Any]) -> str:
    """Render safe content with generated source positions for traceability."""

    assert_safe_model_input(payload)
    theory_lines = [
        f"s{index}: {text}" for index, text in enumerate(payload["theory"], start=1)
    ]
    return "THEORY\n" + "\n".join(theory_lines) + f"\nQUESTION\nq1: {payload['question']}"


def fixed_sample(
    examples: Sequence[Example],
    count: int,
    seed: int,
    max_questions_per_theory: int | None = 1,
) -> list[Example]:
    if count < 0:
        raise ValueError("sample count must be non-negative")
    if count == 0:
        return []
    indices = list(range(len(examples)))
    random.Random(seed).shuffle(indices)
    selected: list[Example] = []
    theory_counts: Counter[str] = Counter()
    for index in indices:
        item = examples[index]
        key = _theory_key(item)
        if (
            max_questions_per_theory is not None
            and theory_counts[key] >= max_questions_per_theory
        ):
            continue
        selected.append(item)
        theory_counts[key] += 1
        if len(selected) == count:
            return selected
    raise ValueError(
        f"requested {count} examples but only {len(selected)} satisfy "
        f"max_questions_per_theory={max_questions_per_theory}"
    )


def balanced_sample(
    examples: Sequence[Example],
    count: int,
    seed: int,
    max_questions_per_theory: int | None = 1,
) -> list[Example]:
    """Select a deterministic, label-balanced sample with broad surface coverage.

    The selector is evaluator-side: labels and depths affect only which IDs are
    selected. Model content still goes exclusively through ``build_model_input``.
    Ties are broken by a stable hash rather than dataset iteration order.
    """

    if count < 0:
        raise ValueError("sample count must be non-negative")
    if count == 0:
        return []

    label_order = ("True", "False", "Unknown")
    per_label = count // len(label_order)
    quotas = {label: per_label for label in label_order}
    for label in ("Unknown", "True", "False")[: count % len(label_order)]:
        quotas[label] += 1

    candidates: dict[str, list[Example]] = {}
    for label in label_order:
        rows = [example for example in examples if example.gold_label == label]
        if len(rows) < quotas[label]:
            raise ValueError(
                f"balanced sample needs {quotas[label]} {label} examples; "
                f"dataset has {len(rows)}"
            )
        candidates[label] = sorted(rows, key=lambda item: _stable_sample_key(item, seed))

    selected: list[Example] = []
    covered: set[tuple[str, Any]] = set()
    theory_counts: Counter[str] = Counter()
    remaining = dict(quotas)
    while len(selected) < count:
        for label in label_order:
            if remaining[label] == 0:
                continue
            eligible = [
                item
                for item in candidates[label]
                if max_questions_per_theory is None
                or theory_counts[_theory_key(item)] < max_questions_per_theory
            ]
            if not eligible:
                raise ValueError(
                    "balanced sample cannot satisfy label quotas with "
                    f"max_questions_per_theory={max_questions_per_theory}"
                )
            best = max(
                eligible,
                key=lambda item: (
                    len(_coverage_keys(item) - covered),
                    -int(_stable_sample_key(item, seed), 16),
                ),
            )
            candidates[label].remove(best)
            selected.append(best)
            covered.update(_coverage_keys(best))
            theory_counts[_theory_key(best)] += 1
            remaining[label] -= 1
    return selected


def exclude_example_ids(
    examples: Sequence[Example], excluded_ids: Iterable[str]
) -> list[Example]:
    """Return a stable question-level filtered copy."""

    excluded = set(excluded_ids)
    return [example for example in examples if example.example_id not in excluded]


def exclude_quarantined_examples(
    examples: Sequence[Example],
    excluded_ids: Iterable[str],
    excluded_theory_ids: Iterable[str],
) -> list[Example]:
    """Apply both observed-question and exposed-theory quarantine gates."""

    question_ids = set(excluded_ids)
    theory_ids = set(excluded_theory_ids)
    return [
        example
        for example in examples
        if example.example_id not in question_ids and example.theory_id not in theory_ids
    ]


def example_surface_features(example: Example) -> dict[str, Any]:
    """Return auditable natural-language coverage features for sample manifests."""

    combined = " ".join(
        [sentence.text for sentence in example.theory] + [example.question.text]
    )
    return {
        "explicit_negation": bool(
            re.search(r"\b(?:not|no|never)\b", combined, flags=re.IGNORECASE)
        ),
        "question_explicit_negation": bool(
            re.search(r"\b(?:not|no|never)\b", example.question.text, flags=re.IGNORECASE)
        ),
        "question_relation_surface": _question_relation_surface(example.question.text),
    }


def dataset_fingerprint(examples: Iterable[Example]) -> str:
    digest = hashlib.sha256()
    for example in examples:
        digest.update(example.example_id.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_jsonl_schema(path: str | Path, limit: int = 5) -> dict[str, Any]:
    """Return observed keys and types without assigning semantics to unknown data."""

    records = _read_jsonl(Path(path), limit=limit)
    if not records:
        raise DatasetSchemaError(f"dataset is empty: {path}")
    observed: dict[str, set[str]] = {}
    for record in records:
        for key, value in record.items():
            observed.setdefault(key, set()).add(type(value).__name__)
    return {
        "path": str(Path(path).resolve()),
        "records_inspected": len(records),
        "root_fields": {key: sorted(types) for key, types in sorted(observed.items())},
        "detected_profile": detect_profile(records[0]),
    }


def inspect_proofwriter_dataset(path: str | Path) -> dict[str, Any]:
    """Inspect a complete documented ProofWriter main split without exposing proofs."""

    dataset_path = Path(path)
    root_types: dict[str, set[str]] = {}
    triple_fields: set[str] = set()
    rule_fields: set[str] = set()
    question_fields: set[str] = set()
    label_counts: Counter[str] = Counter()
    depth_counts: Counter[str] = Counter()
    records = 0
    questions = 0
    theory_mismatches = 0
    count_mismatches = 0
    profiles: set[str] = set()
    theory_examples: list[str] = []
    question_examples: list[str] = []
    documented_root_fields = {
        "id",
        "maxD",
        "NFact",
        "NRule",
        "theory",
        "triples",
        "rules",
        "questions",
        "allProofs",
        "proofDetails",
    }
    extra_root_fields: set[str] = set()

    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetSchemaError(
                    f"invalid JSON at {dataset_path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise DatasetSchemaError(
                    f"record at {dataset_path}:{line_number} is not an object"
                )
            profile = detect_profile(record)
            profiles.add(profile)
            if profile != "proofwriter_nested":
                raise DatasetSchemaError(
                    f"record {line_number} is {profile}, expected proofwriter_nested"
                )
            records += 1
            extra_root_fields.update(set(record) - documented_root_fields)
            for key, value in record.items():
                root_types.setdefault(key, set()).add(type(value).__name__)

            triples = record["triples"]
            rules = record["rules"]
            nested_questions = record["questions"]
            if not all(isinstance(value, dict) for value in (triples, rules, nested_questions)):
                raise DatasetSchemaError(
                    f"record {line_number} triples/rules/questions must be objects"
                )
            texts: list[str] = []
            for item in triples.values():
                if not isinstance(item, dict):
                    raise DatasetSchemaError(f"record {line_number} has a non-object triple")
                triple_fields.update(item)
                if not isinstance(item.get("text"), str):
                    raise DatasetSchemaError(f"record {line_number} triple text is not a string")
                texts.append(item["text"].strip())
            for item in rules.values():
                if not isinstance(item, dict):
                    raise DatasetSchemaError(f"record {line_number} has a non-object rule")
                rule_fields.update(item)
                if not isinstance(item.get("text"), str):
                    raise DatasetSchemaError(f"record {line_number} rule text is not a string")
                texts.append(item["text"].strip())
            theory = record.get("theory")
            if not isinstance(theory, str):
                raise DatasetSchemaError(f"record {line_number} theory is not a string")
            if theory.strip() != " ".join(texts):
                theory_mismatches += 1
            if record.get("NFact") != len(triples) or record.get("NRule") != len(rules):
                count_mismatches += 1
            if len(theory_examples) < 2:
                theory_examples.append(theory[:300])

            for item in nested_questions.values():
                if not isinstance(item, dict):
                    raise DatasetSchemaError(f"record {line_number} has a non-object question")
                question_fields.update(item)
                if not isinstance(item.get("question"), str):
                    raise DatasetSchemaError(f"record {line_number} question text is not a string")
                label = _label(item.get("answer"), f"record[{line_number}].answer")
                depth = _depth(item.get("QDep"), f"record[{line_number}].QDep")
                label_counts[label] += 1
                depth_counts["null" if depth is None else str(depth)] += 1
                questions += 1
                if len(question_examples) < 5:
                    question_examples.append(item["question"])

    assumption = _assumption_from_path(dataset_path)
    return {
        "path": str(dataset_path.resolve()),
        "sha256": file_sha256(dataset_path),
        "source_profile": sorted(profiles),
        "record_count": records,
        "question_count": questions,
        "root_fields": {key: sorted(types) for key, types in sorted(root_types.items())},
        "triple_item_fields": sorted(triple_fields),
        "rule_item_fields": sorted(rule_fields),
        "question_item_fields": sorted(question_fields),
        "theory_format": "root string; adapter uses ordered triples[*].text + rules[*].text",
        "question_format": "questions object keyed by Q*, each with a question string",
        "theory_examples": theory_examples,
        "question_examples": question_examples,
        "label_distribution": dict(sorted(label_counts.items())),
        "depth_distribution": dict(
            sorted(depth_counts.items(), key=lambda item: (item[0] == "null", item[0]))
        ),
        "world_assumption": assumption,
        "assumption_consistent": assumption != "OWA" or label_counts["Unknown"] > 0,
        "theory_text_mismatch_count": theory_mismatches,
        "declared_count_mismatch_count": count_mismatches,
        "unexpected_root_fields": sorted(extra_root_fields),
        "model_input_whitelist": sorted(MODEL_INPUT_FIELDS),
        "evaluator_only_fields": sorted(FORBIDDEN_DATASET_FIELDS),
    }


def detect_profile(record: Mapping[str, Any]) -> str:
    keys = set(record)
    if {"example_id", "theory", "question", "gold_label"} <= keys:
        return "vgcf_flat_v1"
    if {"id", "triples", "rules", "questions"} <= keys:
        return "proofwriter_nested"
    if {"id", "theory", "question", "answer"} <= keys:
        return "proofwriter_flat"
    return "unknown"


def load_examples(path: str | Path) -> list[Example]:
    dataset_path = Path(path)
    records = _read_jsonl(dataset_path)
    if not records:
        raise DatasetSchemaError(f"dataset is empty: {path}")
    profile = detect_profile(records[0])
    if profile == "unknown":
        raise DatasetSchemaError(
            "unknown dataset schema; inspect it before adding an adapter. "
            f"Observed root fields: {sorted(records[0])}"
        )
    if any(detect_profile(record) != profile for record in records):
        raise DatasetSchemaError("mixed dataset profiles in one JSONL file")
    if profile == "vgcf_flat_v1":
        return [_from_fixture(record, index) for index, record in enumerate(records)]
    if profile == "proofwriter_flat":
        return [_from_flat(record, index) for index, record in enumerate(records)]
    examples: list[Example] = []
    source_split = _split_from_path(dataset_path)
    for index, record in enumerate(records):
        examples.extend(_from_nested(record, index, source_split))
    return examples


def _read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        raise DatasetSchemaError(f"dataset does not exist: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetSchemaError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise DatasetSchemaError(f"record at {path}:{line_number} is not an object")
            records.append(record)
            if limit is not None and len(records) >= limit:
                break
    return records


def _source_sentences(theory: Any, path: str) -> tuple[SourceSentence, ...]:
    if isinstance(theory, str):
        texts = [piece.strip() for piece in re.split(r"(?<=[.!?])\s+", theory) if piece.strip()]
    elif isinstance(theory, list) and all(isinstance(item, str) for item in theory):
        texts = [item.strip() for item in theory]
    else:
        raise DatasetSchemaError(f"{path} must be a string or list of strings")
    if not texts or any(not text for text in texts):
        raise DatasetSchemaError(f"{path} contains no usable theory sentences")
    return tuple(SourceSentence(f"s{index}", text) for index, text in enumerate(texts, 1))


def _label(value: Any, path: str) -> str:
    if value is True:
        return "True"
    if value is False:
        return "False"
    if isinstance(value, str):
        normalized = value.strip().lower()
        mapping = {"true": "True", "false": "False", "unknown": "Unknown"}
        if normalized in mapping:
            return mapping[normalized]
    raise DatasetSchemaError(f"{path} must be True, False, or Unknown")


def _depth(value: Any, path: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DatasetSchemaError(f"{path} must be a non-negative integer or null")
    return value


def _from_fixture(record: Mapping[str, Any], index: int) -> Example:
    allowed = {"example_id", "depth", "theory", "question", "gold_label"}
    unknown = set(record) - allowed
    if unknown:
        raise DatasetSchemaError(f"fixture record {index} has unknown fields: {sorted(unknown)}")
    for field in {"example_id", "theory", "question", "gold_label"}:
        if field not in record:
            raise DatasetSchemaError(f"fixture record {index} missing {field}")
    if not isinstance(record["example_id"], str) or not isinstance(record["question"], str):
        raise DatasetSchemaError(f"fixture record {index} has invalid id/question")
    return Example(
        example_id=record["example_id"],
        depth=_depth(record.get("depth"), f"record[{index}].depth"),
        theory=_source_sentences(record["theory"], f"record[{index}].theory"),
        question=SourceSentence("q1", record["question"].strip()),
        gold_label=_label(record["gold_label"], f"record[{index}].gold_label"),
        source_profile="vgcf_flat_v1",
    )


def _from_flat(record: Mapping[str, Any], index: int) -> Example:
    """Adapt only explicitly named flat fields; all other metadata is ignored."""

    if not isinstance(record["id"], str) or not isinstance(record["question"], str):
        raise DatasetSchemaError(f"flat record {index} has invalid id/question")
    depth_value = record.get("depth", record.get("QDep"))
    return Example(
        example_id=record["id"],
        depth=_depth(depth_value, f"record[{index}].depth"),
        theory=_source_sentences(record["theory"], f"record[{index}].theory"),
        question=SourceSentence("q1", record["question"].strip()),
        gold_label=_label(record["answer"], f"record[{index}].answer"),
        source_profile="proofwriter_flat",
    )


def _from_nested(
    record: Mapping[str, Any], index: int, source_split: str
) -> list[Example]:
    """Adapt the documented ProofWriter nested story/question organization.

    Only ``text``/``question`` reach the model. Logic representations and proof
    metadata, if present, are deliberately never read here.
    """

    story_id = record["id"]
    triples, rules, questions = record["triples"], record["rules"], record["questions"]
    if not isinstance(story_id, str):
        raise DatasetSchemaError(f"nested record {index}.id must be a string")
    if not all(isinstance(value, dict) for value in (triples, rules, questions)):
        raise DatasetSchemaError(f"nested record {index} triples/rules/questions must be objects")
    texts: list[str] = []
    for group_name, group in (("triples", triples), ("rules", rules)):
        for key, item in group.items():
            if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                raise DatasetSchemaError(
                    f"nested record {index}.{group_name}.{key}.text must be a string"
                )
            texts.append(item["text"].strip())
    sources = _source_sentences(texts, f"record[{index}].theory")
    result: list[Example] = []
    for question_id, item in questions.items():
        if not isinstance(item, dict):
            raise DatasetSchemaError(f"nested question {story_id}/{question_id} is not an object")
        if not isinstance(item.get("question"), str) or "answer" not in item:
            raise DatasetSchemaError(
                f"nested question {story_id}/{question_id} requires question and answer"
            )
        depth_value = item.get("QDep", item.get("depth"))
        result.append(
            Example(
                example_id=f"{story_id}:{question_id}",
                depth=_depth(depth_value, f"{story_id}/{question_id}.QDep"),
                theory=sources,
                question=SourceSentence("q1", item["question"].strip()),
                gold_label=_label(item["answer"], f"{story_id}/{question_id}.answer"),
                source_profile="proofwriter_nested",
                theory_id=story_id,
                source_split=source_split,
            )
        )
    return result


def _stable_sample_key(example: Example, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{example.example_id}".encode("utf-8")).hexdigest()


def _theory_key(example: Example) -> str:
    if example.theory_id:
        return example.theory_id
    encoded = json.dumps(
        [sentence.text for sentence in example.theory],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _split_from_path(path: Path) -> str:
    name = path.name.lower()
    for split in ("meta-train", "meta-dev", "meta-test"):
        if name == f"{split}.jsonl":
            return split
    return "unknown"


def _coverage_keys(example: Example) -> set[tuple[str, Any]]:
    features = example_surface_features(example)
    return {
        ("depth", example.depth),
        ("theory_explicit_negation", features["explicit_negation"]),
        ("question_explicit_negation", features["question_explicit_negation"]),
        ("question_relation_surface", features["question_relation_surface"]),
    }


def _question_relation_surface(question: str) -> str:
    normalized = question.strip().lower()
    if re.search(r"\b(?:is|are) (?:not )?[a-z0-9_-]+[.?]?$", normalized):
        return "unary"
    return "binary"


def _assumption_from_path(path: Path) -> str:
    parts = {part.upper() for part in path.parts}
    if "OWA" in parts:
        return "OWA"
    if "CWA" in parts:
        return "CWA"
    return "unknown"
