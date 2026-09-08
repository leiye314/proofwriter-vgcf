"""Strict and lenient parsing of model responses."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .errors import SchemaError
from .ir_v2 import parse_v2_formalization
from .schema import Formalization

LABEL_PATTERN = re.compile(r"\b(True|False|Unknown)\b", re.IGNORECASE)
_LABELS = frozenset({"True", "False", "Unknown"})
_FINAL_LABEL_ANCHOR = "FINAL_LABEL_FOR_ORIGINAL_QUESTION:"
_FINAL_LABEL_LINE = re.compile(
    r"FINAL_LABEL_FOR_ORIGINAL_QUESTION: (True|False|Unknown)"
)


@dataclass(frozen=True, slots=True)
class AnchoredLabelResponse:
    """A visible rationale and its strictly anchored original-question label."""

    section: str
    rationale: str
    label: str


def parse_label(text: str) -> str:
    matches = LABEL_PATTERN.findall(text)
    if not matches:
        raise SchemaError("model response contains no True/False/Unknown label")
    normalized = matches[-1].lower()
    return {"true": "True", "false": "False", "unknown": "Unknown"}[normalized]


def parse_cot_response(text: str) -> AnchoredLabelResponse:
    """Parse the VGCF-2.3 visible CoT contract without a regex fallback."""

    return _parse_anchored_label_response(text, section="REASONING")


def parse_cot_refine_response(text: str) -> AnchoredLabelResponse:
    """Parse the VGCF-2.3 visible CoT-Refine review contract."""

    return _parse_anchored_label_response(text, section="REVIEW")


def parse_cot_label(text: str) -> str:
    return parse_cot_response(text).label


def parse_cot_refine_label(text: str) -> str:
    return parse_cot_refine_response(text).label


def _parse_anchored_label_response(
    text: str, *, section: str
) -> AnchoredLabelResponse:
    if not isinstance(text, str):
        raise SchemaError(f"{section} response must be text")
    lines = text.splitlines()
    nonempty = [index for index, line in enumerate(lines) if line.strip()]
    if not nonempty:
        raise SchemaError(f"{section} response is empty")

    anchor_occurrences = re.findall(
        re.escape(_FINAL_LABEL_ANCHOR), text, flags=re.IGNORECASE
    )
    if len(anchor_occurrences) != 1:
        qualifier = "missing" if not anchor_occurrences else "repeated"
        raise SchemaError(
            f"{section} response has {qualifier} {_FINAL_LABEL_ANCHOR} anchor"
        )

    anchored_lines = [
        index
        for index, line in enumerate(lines)
        if line.strip().lower().startswith(_FINAL_LABEL_ANCHOR.lower())
    ]
    if len(anchored_lines) != 1:
        raise SchemaError(f"{section} response must contain exactly one final-label line")
    anchor_index = anchored_lines[0]
    match = _FINAL_LABEL_LINE.fullmatch(lines[anchor_index].strip())
    if match is None or match.group(1) not in _LABELS:
        raise SchemaError(
            f"{section} final label must be exactly True, False, or Unknown"
        )
    if anchor_index != nonempty[-1]:
        raise SchemaError(f"{section} final-label line must be the last non-empty line")

    header = f"{section}:"
    if lines[nonempty[0]].strip() != header:
        raise SchemaError(f"{section} response must start with {header}")
    if nonempty[0] >= anchor_index:
        raise SchemaError(f"{section} response is missing visible object-level rationale")
    rationale = "\n".join(lines[nonempty[0] + 1 : anchor_index]).strip()
    if not rationale:
        raise SchemaError(f"{section} response is missing visible object-level rationale")
    return AnchoredLabelResponse(section, rationale, match.group(1))


def parse_strict_formalization(text: str) -> Formalization:
    return parse_v2_formalization(text)


def parse_lenient_formalization(text: str) -> Formalization:
    """Plain uses the same v2 IR; only one complete Markdown fence is tolerated."""

    return parse_v2_formalization(text)


def _balanced_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None
