"""Fail-closed dataset, sample, quarantine, and final-test protocol rules."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .data import Example, ModelExample, file_sha256
from .errors import DatasetSchemaError

PHASES = frozenset(
    {"software_validation", "unblinded_pilot", "train", "dev", "final_test"}
)
KNOWN_DATASET_SHA256 = {
    "meta-train": "d0501d29506159e9b59f3cb9dcff10b4eb44f4d94dd52f9ee0bf93e5054a262f",
    "meta-dev": "1a2e13c9e57c535da46c22622cc46185fd1a13ce49522660eda1eed38509db98",
    "meta-test": "c09fad796aaf546d6fcbfc77ecf91f935ffed3c936c2b0e96f4aa57211fad842",
}
FINAL_TEST_ARM_ENV = "VGCF_FINAL_TEST_ARM"
FINAL_TEST_ARM_VALUE = "VGCF2_1_FINAL_TEST"
FINAL_TEST_ARM_VALUE_V2_2 = "VGCF2_2_FINAL_TEST"
FINAL_TEST_ARM_VALUE_V2_3 = "VGCF2_3_FINAL_TEST"
FINAL_TEST_ARM_VALUE_V2_3_1 = "VGCF2_3_1_FINAL_TEST"
FINAL_LOCK_SCHEMA_BY_PROTOCOL = {
    "VGCF-2.1": 1,
    "VGCF-2.2": 2,
    "VGCF-2.3": 3,
    "VGCF-2.3.1": 5,
}


@dataclass(frozen=True, slots=True)
class DatasetIdentity:
    split: str
    sha256: str


@dataclass(frozen=True, slots=True)
class TheoryQuarantine:
    theory_ids: frozenset[str]
    excluded_question_count: int
    observed_example_count: int
    dataset_sha256: str


def identify_dataset(dataset_path: str | Path) -> DatasetIdentity:
    """Identify a release split by bytes, never by filename alone."""

    path = Path(dataset_path)
    if not path.exists():
        raise DatasetSchemaError(f"dataset does not exist for identity audit: {path}")
    digest = file_sha256(path)
    matches = [split for split, expected in KNOWN_DATASET_SHA256.items() if digest == expected]
    if matches:
        return DatasetIdentity(matches[0], digest)
    split_names = {f"{split}.jsonl" for split in KNOWN_DATASET_SHA256}
    if path.name.lower() in split_names:
        raise DatasetSchemaError(
            f"known ProofWriter split filename {path.name!r} has unrecognized SHA-256 {digest}"
        )
    return DatasetIdentity("unknown", digest)


def audit_dataset_for_phase(
    phase: str,
    dataset_path: str | Path,
    *,
    final_test_armed: bool = False,
    config_only: bool = False,
) -> DatasetIdentity:
    """Verify byte identity and enforce phase isolation.

    ``config_only`` permits parsing a final-test config while keeping execution
    locked. It never permits another phase to touch meta-test.
    """

    if phase not in PHASES:
        raise ValueError(f"phase must be one of {sorted(PHASES)}")
    identity = identify_dataset(dataset_path)
    if identity.split == "meta-test" and not (
        phase == "final_test" and (final_test_armed or config_only)
    ):
        raise DatasetSchemaError(
            f"{phase} phase rejected the meta-test dataset identity; final_test must be explicitly armed"
        )
    expected = {"train": "meta-train", "dev": "meta-dev", "final_test": "meta-test"}.get(
        phase
    )
    if expected is not None and identity.split != expected:
        raise DatasetSchemaError(
            f"{phase} phase requires the known {expected} SHA-256 identity; "
            f"received split={identity.split!r} sha256={identity.sha256}"
        )
    if phase == "final_test" and not (final_test_armed or config_only):
        raise DatasetSchemaError(
            f"final_test is locked; set {FINAL_TEST_ARM_ENV} only after the protocol is frozen"
        )
    return identity


def audit_model_example(
    phase: str, example: ModelExample, *, final_test_armed: bool = False
) -> None:
    """Reject a misrouted sample at the last boundary before prompt creation."""

    if example.source_split == "meta-test" and not (
        phase == "final_test" and final_test_armed
    ):
        raise DatasetSchemaError(
            f"{phase} request rejected meta-test sample provenance without an armed final_test"
        )
    expected = {"train": "meta-train", "dev": "meta-dev", "final_test": "meta-test"}.get(
        phase
    )
    if expected is not None and example.source_split != expected:
        raise DatasetSchemaError(
            f"{phase} request rejected: sample provenance is {example.source_split!r}, "
            f"expected {expected!r}"
        )
    if phase == "final_test" and not final_test_armed:
        raise DatasetSchemaError("final_test request rejected because the runner is not armed")


def load_quarantine(path: str | Path) -> dict[str, tuple[str, ...]]:
    """Load the observed question-level quarantine with source provenance."""

    value: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise DatasetSchemaError("quarantine file must be a JSON object")
    if value.get("schema_version") != 1:
        raise DatasetSchemaError("quarantine schema_version must be 1")
    entries = value.get("entries")
    if not isinstance(entries, list):
        raise DatasetSchemaError("quarantine.entries must be an array")
    result: dict[str, tuple[str, ...]] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise DatasetSchemaError(f"quarantine.entries[{index}] must be an object")
        if set(entry) != {"example_id", "sources"}:
            raise DatasetSchemaError(
                f"quarantine.entries[{index}] must contain only example_id and sources"
            )
        example_id = entry["example_id"]
        sources = entry["sources"]
        if not isinstance(example_id, str) or not example_id:
            raise DatasetSchemaError(f"quarantine.entries[{index}].example_id is invalid")
        if (
            not isinstance(sources, list)
            or not sources
            or not all(isinstance(source, str) and source for source in sources)
        ):
            raise DatasetSchemaError(f"quarantine.entries[{index}].sources is invalid")
        if example_id in result:
            raise DatasetSchemaError(f"duplicate quarantine ID {example_id}")
        result[example_id] = tuple(sources)
    if value.get("id_count") != len(result):
        raise DatasetSchemaError("quarantine id_count does not match its entries")
    return result


def load_theory_quarantine(path: str | Path) -> TheoryQuarantine:
    value: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or value.get("schema_version") != 1:
        raise DatasetSchemaError("theory quarantine must be a schema_version 1 object")
    entries = value.get("entries")
    if not isinstance(entries, list):
        raise DatasetSchemaError("theory quarantine entries must be an array")
    theory_ids: list[str] = []
    excluded_total = 0
    observed_total = 0
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise DatasetSchemaError(f"theory quarantine entry {index} must be an object")
        theory_id = entry.get("theory_id")
        observed = entry.get("observed_example_ids")
        excluded = entry.get("excluded_question_count")
        if not isinstance(theory_id, str) or not theory_id:
            raise DatasetSchemaError(f"theory quarantine entry {index} has invalid theory_id")
        if not isinstance(observed, list) or not all(
            isinstance(item, str) and item.startswith(theory_id + ":") for item in observed
        ):
            raise DatasetSchemaError(f"theory quarantine entry {index} has invalid observed IDs")
        if not isinstance(excluded, int) or excluded < len(observed):
            raise DatasetSchemaError(f"theory quarantine entry {index} has invalid question count")
        theory_ids.append(theory_id)
        observed_total += len(observed)
        excluded_total += excluded
    if len(set(theory_ids)) != len(theory_ids):
        raise DatasetSchemaError("duplicate theory quarantine ID")
    declared = (
        value.get("theory_count"),
        value.get("observed_example_count"),
        value.get("excluded_question_count"),
    )
    actual = (len(theory_ids), observed_total, excluded_total)
    if declared != actual:
        raise DatasetSchemaError(
            f"theory quarantine declared counts {declared!r} do not match {actual!r}"
        )
    dataset_sha256 = value.get("dataset_sha256")
    if dataset_sha256 != KNOWN_DATASET_SHA256["meta-test"]:
        raise DatasetSchemaError("theory quarantine is not bound to the known meta-test hash")
    return TheoryQuarantine(
        frozenset(theory_ids), excluded_total, observed_total, dataset_sha256
    )


def load_sample_manifest(path: str | Path) -> Mapping[str, Any]:
    value: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        not isinstance(value, Mapping)
        or value.get("schema_version") not in {1, 2}
    ):
        raise DatasetSchemaError(
            "sample manifest must be a schema_version 1 or 2 object"
        )
    if value.get("schema_version") == 1:
        if (
            value.get("dataset_split") != "meta-dev"
            or value.get("dataset_sha256") != KNOWN_DATASET_SHA256["meta-dev"]
        ):
            raise DatasetSchemaError(
                "schema-version 1 sample manifest must bind known meta-dev"
            )
    elif (
        value.get("protocol") != "VGCF-2.3.1"
        or value.get("dataset_split") != "meta-test"
        or value.get("dataset_sha256") != KNOWN_DATASET_SHA256["meta-test"]
    ):
        raise DatasetSchemaError(
            "schema-version 2 sample manifest must bind VGCF-2.3.1 meta-test"
        )
    sets = value.get("sets")
    if not isinstance(sets, Mapping) or not sets:
        raise DatasetSchemaError("sample manifest sets must be a non-empty object")
    for name, sample in sets.items():
        if not isinstance(name, str) or not isinstance(sample, Mapping):
            raise DatasetSchemaError("sample manifest contains an invalid set")
        ids = sample.get("example_ids")
        if not isinstance(ids, list) or not all(
            isinstance(item, str) and item for item in ids
        ):
            raise DatasetSchemaError(f"sample set {name!r} has invalid example_ids")
        if sample.get("example_count") != len(ids) or len(set(ids)) != len(ids):
            raise DatasetSchemaError(f"sample set {name!r} count/uniqueness check failed")
        if value.get("schema_version") == 2:
            entries = sample.get("entries")
            if not isinstance(entries, list) or len(entries) != len(ids):
                raise DatasetSchemaError(
                    f"final sample set {name!r} has invalid frozen entries"
                )
            entry_ids = [
                item.get("example_id") if isinstance(item, Mapping) else None
                for item in entries
            ]
            theory_ids = [
                item.get("theory_id") if isinstance(item, Mapping) else None
                for item in entries
            ]
            if entry_ids != ids or not all(
                isinstance(item, str) and item for item in theory_ids
            ):
                raise DatasetSchemaError(
                    f"final sample set {name!r} entry order/identity mismatch"
                )
            if sample.get("theory_count") != len(set(theory_ids)):
                raise DatasetSchemaError(
                    f"final sample set {name!r} theory count mismatch"
                )
    return value


def select_manifest_examples(
    examples: Sequence[Example], manifest: Mapping[str, Any], sample_set: str
) -> list[Example]:
    sets = manifest["sets"]
    if sample_set not in sets:
        raise DatasetSchemaError(f"unknown sample set {sample_set!r}")
    wanted = list(sets[sample_set]["example_ids"])
    by_id = {example.example_id: example for example in examples}
    missing = [example_id for example_id in wanted if example_id not in by_id]
    if missing:
        raise DatasetSchemaError(f"sample manifest IDs missing from dataset: {missing[:10]}")
    return [by_id[example_id] for example_id in wanted]


def audit_sample_disjointness(
    manifest: Mapping[str, Any], examples: Sequence[Example]
) -> dict[str, Any]:
    by_id = {example.example_id: example for example in examples}
    sets = manifest["sets"]
    names = list(sets)
    example_overlaps: list[dict[str, Any]] = []
    theory_overlaps: list[dict[str, Any]] = []
    for left_index, left in enumerate(names):
        left_ids = set(sets[left]["example_ids"])
        for right in names[left_index + 1 :]:
            right_ids = set(sets[right]["example_ids"])
            shared_ids = sorted(left_ids & right_ids)
            left_theories = {by_id[item].theory_id for item in left_ids if item in by_id}
            right_theories = {by_id[item].theory_id for item in right_ids if item in by_id}
            shared_theories = sorted(left_theories & right_theories)
            if shared_ids:
                example_overlaps.append({"sets": [left, right], "example_ids": shared_ids})
            if shared_theories:
                theory_overlaps.append({"sets": [left, right], "theory_ids": shared_theories})
    missing = sorted(
        item
        for sample in sets.values()
        for item in sample["example_ids"]
        if item not in by_id
    )
    report = {
        "status": "pass" if not (missing or example_overlaps or theory_overlaps) else "fail",
        "set_counts": {name: sets[name]["example_count"] for name in names},
        "missing_example_ids": missing,
        "example_overlaps": example_overlaps,
        "theory_overlaps": theory_overlaps,
    }
    if report["status"] != "pass":
        raise DatasetSchemaError(f"sample disjointness audit failed: {report}")
    return report


def arm_final_test(
    *,
    config_hash: str,
    protocol_version: str,
    dataset_path: str | Path,
    prompt_directory: str | Path,
    quarantine_path: str | Path | None,
    theory_quarantine_path: str | Path | None,
    lock_path: str | Path | None,
    sample_manifest_path: str | Path | None = None,
    expected_sample_manifest_sha256: str | None = None,
) -> Mapping[str, Any]:
    """Validate every final-test lock before a client can be constructed."""

    if (
        quarantine_path is None
        or theory_quarantine_path is None
        or lock_path is None
    ):
        raise DatasetSchemaError(
            "final_test requires question/theory quarantine and a protocol lock"
        )
    lock: Any = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    if (
        not isinstance(lock, Mapping)
        or type(lock.get("schema_version")) is not int
        or lock.get("schema_version") not in {1, 2, 3, 4, 5}
    ):
        raise DatasetSchemaError(
            "final-test lock must be a schema_version 1, 2, 3, 4, or 5 object"
        )
    lock_schema = lock["schema_version"]
    if lock_schema == 4:
        raise DatasetSchemaError(
            "schema-version 4 Final lock is historical/superseded and has no "
            "Final activation authority; VGCF-2.3.1 requires schema version 5"
        )
    expected_schema = FINAL_LOCK_SCHEMA_BY_PROTOCOL.get(protocol_version)
    if expected_schema is None:
        raise DatasetSchemaError(
            f"unsupported final-test config protocol_version {protocol_version!r}"
        )
    if lock_schema != expected_schema:
        raise DatasetSchemaError(
            "final-test protocol/lock schema mismatch: "
            f"config {protocol_version} requires schema version {expected_schema}, "
            f"received {lock_schema}"
        )
    required_arm_value = {
        1: FINAL_TEST_ARM_VALUE,
        2: FINAL_TEST_ARM_VALUE_V2_2,
        3: FINAL_TEST_ARM_VALUE_V2_3,
        5: FINAL_TEST_ARM_VALUE_V2_3_1,
    }[lock_schema]
    if (
        lock_schema in {2, 3, 5}
        and lock.get("arm_value") != required_arm_value
    ):
        raise DatasetSchemaError(
            f"{lock.get('protocol', 'versioned')} final-test lock has an invalid arm value"
        )
    if os.environ.get(FINAL_TEST_ARM_ENV) != required_arm_value:
        raise DatasetSchemaError(
            f"final_test is not armed; require {FINAL_TEST_ARM_ENV}={required_arm_value}"
        )
    identity = audit_dataset_for_phase("final_test", dataset_path, final_test_armed=True)
    question_quarantine = load_quarantine(quarantine_path)
    theory_quarantine = load_theory_quarantine(theory_quarantine_path)
    if {item.split(":", 1)[0] for item in question_quarantine} != set(
        theory_quarantine.theory_ids
    ):
        raise DatasetSchemaError("question and theory quarantine artifacts disagree")
    if lock.get("dataset_sha256") != identity.sha256:
        raise DatasetSchemaError("final-test lock dataset hash mismatch")
    if lock.get("config_sha256") != config_hash:
        raise DatasetSchemaError("final-test config hash is not frozen in the lock")
    expected_quarantine_hashes = lock.get("quarantine_sha256")
    actual_quarantine_hashes = {
        "questions": file_sha256(quarantine_path),
        "theories": file_sha256(theory_quarantine_path),
    }
    if expected_quarantine_hashes != actual_quarantine_hashes:
        raise DatasetSchemaError("final-test quarantine hashes are not frozen")
    expected_prompts = lock.get("prompt_sha256")
    if not isinstance(expected_prompts, Mapping) or not expected_prompts:
        raise DatasetSchemaError("final-test lock has no prompt hashes")
    actual_prompts = {
        name: file_sha256(Path(prompt_directory) / name) for name in expected_prompts
    }
    if dict(expected_prompts) != actual_prompts:
        raise DatasetSchemaError("final-test prompt hashes changed after freezing")
    if lock_schema in {2, 3, 5}:
        expected_protocol = {
            2: "VGCF-2.2",
            3: "VGCF-2.3",
            5: "VGCF-2.3.1",
        }[lock_schema]
        if lock.get("protocol") != expected_protocol:
            raise DatasetSchemaError(
                f"schema-version {lock.get('schema_version')} final-test lock must "
                f"freeze {expected_protocol}"
            )
        expected_code = lock.get("inference_contract_sha256")
        if not isinstance(expected_code, Mapping) or not expected_code:
            raise DatasetSchemaError(
                f"{expected_protocol} final-test lock has no code hashes"
            )
        project_root = Path(lock_path).resolve().parent.parent
        actual_code: dict[str, str] = {}
        for relative_name in expected_code:
            if not isinstance(relative_name, str) or Path(relative_name).is_absolute():
                raise DatasetSchemaError("final-test code hash keys must be relative paths")
            candidate = (project_root / relative_name).resolve()
            if project_root not in candidate.parents:
                raise DatasetSchemaError("final-test code hash path escapes the project")
            actual_code[relative_name] = file_sha256(candidate)
        if dict(expected_code) != actual_code:
            raise DatasetSchemaError(
                "final-test inference-contract code changed after freezing"
            )
    if lock_schema == 5:
        if sample_manifest_path is None or expected_sample_manifest_sha256 is None:
            raise DatasetSchemaError(
                "VGCF-2.3.1 requires a config-bound Final sample manifest"
            )
        project_root = Path(lock_path).resolve().parent.parent
        manifest_lock = lock.get("sample_manifest")
        if not isinstance(manifest_lock, Mapping):
            raise DatasetSchemaError("VGCF-2.3.1 lock has no sample manifest")
        locked_manifest_path = _resolve_locked_file(
            project_root, manifest_lock.get("path"), "sample manifest"
        )
        if locked_manifest_path != Path(sample_manifest_path).resolve():
            raise DatasetSchemaError("final-test config and lock manifest paths differ")
        actual_manifest_hash = file_sha256(locked_manifest_path)
        if (
            manifest_lock.get("sha256") != actual_manifest_hash
            or expected_sample_manifest_sha256 != actual_manifest_hash
        ):
            raise DatasetSchemaError("Final-300 manifest hash is not frozen")
        from .final_authorization_v2_3_1 import validate_final_runtime

        validate_final_runtime(project_root, config_hash=config_hash, lock=lock)
    return lock


def _resolve_locked_file(
    project_root: Path, relative_name: Any, description: str
) -> Path:
    if not isinstance(relative_name, str) or Path(relative_name).is_absolute():
        raise DatasetSchemaError(f"{description} lock path must be relative")
    candidate = (project_root / relative_name).resolve()
    if project_root not in candidate.parents or not candidate.is_file():
        raise DatasetSchemaError(
            f"{description} lock path is missing or escapes project"
        )
    return candidate


def verify_contract_lock_v2_3(
    *,
    config_hash: str,
    lock_path: str | Path | None,
    project_root: str | Path,
) -> Mapping[str, Any]:
    """Fail closed before the VGCF-2.3 Contract-12 client is constructed."""

    if lock_path is None:
        raise DatasetSchemaError("VGCF-2.3 Contract-12 requires a protocol lock")
    path = Path(lock_path)
    if not path.exists():
        raise DatasetSchemaError(f"Contract-12 protocol lock is missing: {path}")
    lock: Any = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(lock, Mapping)
        or lock.get("schema_version") != 1
        or lock.get("protocol") != "VGCF-2.3"
        or lock.get("sample_set") != "contract_12"
    ):
        raise DatasetSchemaError("Contract-12 lock schema/protocol/sample mismatch")
    if lock.get("config_sha256") != config_hash:
        raise DatasetSchemaError("Contract-12 config hash is not frozen")
    root = Path(project_root).resolve()
    for group_name in (
        "artifact_sha256",
        "prompt_sha256",
        "inference_contract_sha256",
    ):
        expected = lock.get(group_name)
        if not isinstance(expected, Mapping) or not expected:
            raise DatasetSchemaError(f"Contract-12 lock has no {group_name}")
        actual: dict[str, str] = {}
        for relative_name in expected:
            if not isinstance(relative_name, str) or Path(relative_name).is_absolute():
                raise DatasetSchemaError("Contract-12 hash keys must be relative paths")
            candidate = (root / relative_name).resolve()
            if root not in candidate.parents:
                raise DatasetSchemaError("Contract-12 hash path escapes the project")
            if not candidate.is_file():
                raise DatasetSchemaError(
                    f"Contract-12 locked path is missing: {relative_name}"
                )
            actual[relative_name] = file_sha256(candidate)
        if dict(expected) != actual:
            raise DatasetSchemaError(f"Contract-12 {group_name} changed after freeze")
    return lock


def verify_confirmation_lock(
    *,
    config_hash: str,
    lock_path: str | Path | None,
    project_root: str | Path,
) -> Mapping[str, Any]:
    """Freeze the confirmation protocol before any client is constructed."""

    if lock_path is None:
        raise DatasetSchemaError("VGCF-2.2 confirmation requires a protocol lock")
    path = Path(lock_path)
    if not path.exists():
        raise DatasetSchemaError(f"confirmation protocol lock is missing: {path}")
    lock: Any = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(lock, Mapping)
        or lock.get("schema_version") != 1
        or lock.get("protocol") != "VGCF-2.2"
    ):
        raise DatasetSchemaError("confirmation lock schema/protocol mismatch")
    if lock.get("config_sha256") != config_hash:
        raise DatasetSchemaError("confirmation config hash is not frozen")
    root = Path(project_root).resolve()
    for group_name in (
        "artifact_sha256",
        "prompt_sha256",
        "inference_contract_sha256",
    ):
        expected = lock.get(group_name)
        if not isinstance(expected, Mapping) or not expected:
            raise DatasetSchemaError(f"confirmation lock has no {group_name}")
        actual: dict[str, str] = {}
        for relative_name in expected:
            if not isinstance(relative_name, str) or Path(relative_name).is_absolute():
                raise DatasetSchemaError("confirmation hash keys must be relative paths")
            candidate = (root / relative_name).resolve()
            if root not in candidate.parents:
                raise DatasetSchemaError("confirmation hash path escapes the project")
            actual[relative_name] = file_sha256(candidate)
        if dict(expected) != actual:
            raise DatasetSchemaError(
                f"confirmation {group_name} changed after protocol freeze"
            )
    return lock


def sha256_json_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
