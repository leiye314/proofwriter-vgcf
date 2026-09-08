"""Offline-only Final-300 preparation rules for VGCF-2.3.1."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .data import Example, example_surface_features, file_sha256
from .errors import DatasetSchemaError

FINAL_PROTOCOL = "VGCF-2.3.1"
FINAL_MANIFEST_SCHEMA_VERSION = 2
FINAL_SAMPLE_SET = "final_300"
FINAL_SAMPLE_COUNT = 300
FINAL_SEED = 20260808
FINAL_LABEL_ORDER = ("True", "False", "Unknown")
FINAL_LABEL_QUOTAS = {label: 100 for label in FINAL_LABEL_ORDER}
META_TEST_SHA256 = (
    "c09fad796aaf546d6fcbfc77ecf91f935ffed3c936c2b0e96f4aa57211fad842"
)
SELECTION_ALGORITHM = "label_stratified_seeded_sha256_v1"


def theory_capacity(
    examples: Sequence[Example], max_questions_per_theory: int
) -> int:
    """Return the exact capacity under a per-theory question cap."""

    if max_questions_per_theory < 1:
        raise ValueError("max_questions_per_theory must be at least 1")
    counts = Counter(_required_theory_id(item) for item in examples)
    return sum(min(count, max_questions_per_theory) for count in counts.values())


def enforce_final_theory_capacity(
    examples: Sequence[Example],
    *,
    sample_count: int,
    max_questions_per_theory: int,
) -> dict[str, int]:
    """Fail explicitly when a Final request exceeds quarantine-filtered capacity."""

    unique_theories = len({_required_theory_id(item) for item in examples})
    capacity = theory_capacity(examples, max_questions_per_theory)
    if sample_count > capacity:
        raise ValueError(
            "final preflight theory capacity exceeded: "
            f"sample_count={sample_count}, "
            f"eligible_unique_theory_count={unique_theories}, "
            f"max_questions_per_theory={max_questions_per_theory}, "
            f"capacity={capacity}, shortfall={sample_count - capacity}"
        )
    return {
        "sample_count": sample_count,
        "eligible_unique_theory_count": unique_theories,
        "max_questions_per_theory": max_questions_per_theory,
        "capacity": capacity,
        "shortfall": 0,
    }


def select_final_300(
    examples: Sequence[Example],
    *,
    excluded_example_ids: Iterable[str],
    excluded_theory_ids: Iterable[str],
    seed: int = FINAL_SEED,
) -> list[Example]:
    """Select fixed per-label quotas using only seeded random hash order.

    This selector deliberately has no coverage score and accepts no Dev errors or
    model outputs. Labels form the three evaluator-side strata. Within each
    stratum, SHA-256 of the fixed seed, label, and example ID supplies a stable
    pseudorandom order. The fixed label order applies a global one-theory rule.
    """

    question_quarantine = set(excluded_example_ids)
    theory_quarantine = set(excluded_theory_ids)
    eligible = [
        item
        for item in examples
        if item.example_id not in question_quarantine
        and _required_theory_id(item) not in theory_quarantine
    ]
    selected: list[Example] = []
    selected_theories: set[str] = set()
    for label in FINAL_LABEL_ORDER:
        candidates = sorted(
            (item for item in eligible if item.gold_label == label),
            key=lambda item: (_selection_rank(item, label, seed), item.example_id),
        )
        label_selected = 0
        for item in candidates:
            theory_id = _required_theory_id(item)
            if theory_id in selected_theories:
                continue
            selected.append(item)
            selected_theories.add(theory_id)
            label_selected += 1
            if label_selected == FINAL_LABEL_QUOTAS[label]:
                break
        if label_selected != FINAL_LABEL_QUOTAS[label]:
            raise DatasetSchemaError(
                f"Final-300 cannot satisfy the {label} quota with unique theories: "
                f"selected {label_selected} of {FINAL_LABEL_QUOTAS[label]}"
            )
    if len(selected) != FINAL_SAMPLE_COUNT:
        raise AssertionError("Final-300 selector returned the wrong number of examples")
    return selected


def build_final_manifest(
    examples: Sequence[Example],
    *,
    dataset_sha256: str,
    question_quarantine: Mapping[str, Sequence[str]],
    theory_quarantine_ids: Iterable[str],
    question_quarantine_path: str | Path,
    theory_quarantine_path: str | Path,
    seed: int = FINAL_SEED,
) -> dict[str, Any]:
    """Build the byte-stable Final-300 manifest without any model call."""

    if dataset_sha256 != META_TEST_SHA256:
        raise DatasetSchemaError("Final-300 requires the frozen meta-test identity")
    theory_ids = frozenset(theory_quarantine_ids)
    selected = select_final_300(
        examples,
        excluded_example_ids=question_quarantine,
        excluded_theory_ids=theory_ids,
        seed=seed,
    )
    eligible = [
        item
        for item in examples
        if item.example_id not in question_quarantine
        and _required_theory_id(item) not in theory_ids
    ]
    entries = [_manifest_entry(item) for item in selected]
    ids = [item.example_id for item in selected]
    composition = _composition(selected)
    return {
        "schema_version": FINAL_MANIFEST_SCHEMA_VERSION,
        "protocol": FINAL_PROTOCOL,
        "dataset_split": "meta-test",
        "dataset_sha256": dataset_sha256,
        "quarantine_sha256": {
            "questions": file_sha256(question_quarantine_path),
            "theories": file_sha256(theory_quarantine_path),
        },
        "selection_note": (
            "Evaluator-only label-stratified deterministic random selection. "
            "No coverage-greedy score, Dev error feature, or model output is used."
        ),
        "eligible_pool": {
            "example_count": len(eligible),
            "unique_theory_count": len(
                {_required_theory_id(item) for item in eligible}
            ),
            "question_quarantine_count": len(question_quarantine),
            "theory_quarantine_count": len(theory_ids),
        },
        "sets": {
            FINAL_SAMPLE_SET: {
                "role": "held_out_final_primary_evaluation",
                "selection_algorithm": {
                    "name": SELECTION_ALGORITHM,
                    "seed": seed,
                    "label_order": list(FINAL_LABEL_ORDER),
                    "label_quotas": dict(FINAL_LABEL_QUOTAS),
                    "random_order_key": (
                        "sha256('VGCF-2.3.1\\0' + seed + '\\0' + label + "
                        "+ '\\0' + example_id)"
                    ),
                    "global_theory_exclusion": True,
                    "coverage_greedy": False,
                    "dev_error_features_used": False,
                    "model_outputs_used": False,
                },
                "example_count": len(selected),
                "theory_count": len(
                    {_required_theory_id(item) for item in selected}
                ),
                **composition,
                "selected_ids_sha256": _selected_ids_sha256(ids),
                "example_ids": ids,
                "entries": entries,
            }
        },
    }


def audit_final_manifest(
    manifest: Mapping[str, Any],
    examples: Sequence[Example],
    *,
    dataset_sha256: str,
    question_quarantine: Mapping[str, Sequence[str]],
    theory_quarantine_ids: Iterable[str],
    question_quarantine_path: str | Path,
    theory_quarantine_path: str | Path,
) -> dict[str, Any]:
    """Reproduce Final-300 exactly and report every frozen constraint."""

    expected = build_final_manifest(
        examples,
        dataset_sha256=dataset_sha256,
        question_quarantine=question_quarantine,
        theory_quarantine_ids=theory_quarantine_ids,
        question_quarantine_path=question_quarantine_path,
        theory_quarantine_path=theory_quarantine_path,
        seed=FINAL_SEED,
    )
    sample = manifest.get("sets", {}).get(FINAL_SAMPLE_SET, {})
    ids = sample.get("example_ids", [])
    entries = sample.get("entries", [])
    theory_ids = [
        item.get("theory_id") for item in entries if isinstance(item, Mapping)
    ]
    label_counts = Counter(
        item.get("label") for item in entries if isinstance(item, Mapping)
    )
    question_ids = set(question_quarantine)
    quarantined_theories = set(theory_quarantine_ids)
    checks = {
        "schema_protocol": manifest.get("schema_version")
        == FINAL_MANIFEST_SCHEMA_VERSION
        and manifest.get("protocol") == FINAL_PROTOCOL,
        "dataset_identity": manifest.get("dataset_split") == "meta-test"
        and manifest.get("dataset_sha256") == dataset_sha256 == META_TEST_SHA256,
        "example_count_300": len(ids) == FINAL_SAMPLE_COUNT
        and sample.get("example_count") == FINAL_SAMPLE_COUNT,
        "unique_example_ids": len(set(ids)) == FINAL_SAMPLE_COUNT,
        "unique_theories_300": len(theory_ids) == FINAL_SAMPLE_COUNT
        and len(set(theory_ids)) == FINAL_SAMPLE_COUNT
        and sample.get("theory_count") == FINAL_SAMPLE_COUNT,
        "label_quotas": dict(label_counts) == FINAL_LABEL_QUOTAS,
        "question_quarantine_disjoint": not (set(ids) & question_ids),
        "theory_quarantine_disjoint": not (
            set(theory_ids) & quarantined_theories
        ),
        "selected_ids_hash": sample.get("selected_ids_sha256")
        == _selected_ids_sha256(list(ids)),
        "exact_reproduction": dict(manifest) == expected,
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "composition": {
            key: sample.get(key)
            for key in (
                "label_counts",
                "depth_counts",
                "family_counts",
                "theory_negation_counts",
                "family_by_theory_negation_counts",
                "question_explicit_negation_counts",
            )
        },
        "model_calls_made": 0,
    }


def _selection_rank(example: Example, label: str, seed: int) -> str:
    material = f"{FINAL_PROTOCOL}\0{seed}\0{label}\0{example.example_id}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _selected_ids_sha256(example_ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(example_ids).encode("utf-8")).hexdigest()


def _required_theory_id(example: Example) -> str:
    if not example.theory_id:
        raise DatasetSchemaError(
            f"Final sampling requires a source theory ID: {example.example_id}"
        )
    return example.theory_id


def _theory_family(theory_id: str) -> str:
    if theory_id.startswith("Att"):
        return "Att"
    if theory_id.startswith("Rel"):
        return "Rel"
    raise DatasetSchemaError(f"unrecognized ProofWriter theory family: {theory_id}")


def _theory_negation(theory_id: str) -> str:
    prefix = theory_id.split("-", 1)[0]
    if prefix in {"AttNeg", "RelNeg"}:
        return "Neg"
    if prefix in {"AttNoneg", "RelNoneg"}:
        return "Noneg"
    raise DatasetSchemaError(f"unrecognized ProofWriter negation family: {theory_id}")


def _manifest_entry(example: Example) -> dict[str, Any]:
    theory_id = _required_theory_id(example)
    features = example_surface_features(example)
    return {
        "example_id": example.example_id,
        "theory_id": theory_id,
        "label": example.gold_label,
        "depth": example.depth,
        "family": _theory_family(theory_id),
        "theory_negation": _theory_negation(theory_id),
        "question_explicit_negation": features["question_explicit_negation"],
    }


def _composition(selected: Sequence[Example]) -> dict[str, dict[str, int]]:
    entries = [_manifest_entry(item) for item in selected]
    return {
        "label_counts": _sorted_counter(item["label"] for item in entries),
        "depth_counts": _sorted_counter(str(item["depth"]) for item in entries),
        "family_counts": _sorted_counter(item["family"] for item in entries),
        "theory_negation_counts": _sorted_counter(
            item["theory_negation"] for item in entries
        ),
        "family_by_theory_negation_counts": _sorted_counter(
            f"{item['family']}{item['theory_negation']}" for item in entries
        ),
        "question_explicit_negation_counts": _sorted_counter(
            str(item["question_explicit_negation"]).lower() for item in entries
        ),
    }


def _sorted_counter(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))
