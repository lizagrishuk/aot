"""p-adic branch assignment exactly following the Lab 6 specification (p=3)."""
from __future__ import annotations
from datetime import datetime
import re
from typing import Any

INCIDENT_RESTRICTION = "2026-03-14T18:42:16"
INCIDENT_RESTORED = "2026-03-14T19:12:19"
TIME_RE = re.compile(r"(?<!\d)(\d{1,2}):(\d{2})(?::(\d{2}))?(?!\d)")

SYSTEM_DOCUMENTAL = {
    "system_log", "system_log_excerpt", "incident_report", "timeline", "irrelevant_document"
}
WITNESS = {"statement", "chat"}
RULE_INTERPRETATION = {"policy", "policy_reference", "technical_note", "postmortem_draft"}


def source_level(document: dict[str, Any]) -> int:
    """a0: 0=system/documental, 1=witness, 2=rule/interpretation."""
    kind = document.get("source_type", "")
    if kind in SYSTEM_DOCUMENTAL:
        return 0
    if kind in WITNESS:
        return 1
    if kind in RULE_INTERPRETATION:
        return 2
    # Fail visibly on unclassified source types rather than silently inventing a branch.
    raise ValueError(f"Unmapped source_type for p-adic level a0: {kind!r}")


def status_level(claim: dict[str, Any]) -> int:
    """a1: 0=direct/established, 1=reported or inference, 2=disputed/exception.

    The exception branch is set by an explicit field ``tag == "exception"`` in claims.jsonl
    (conditions of the MOD-4.2-E exception), not by searching the claim text for keywords.
    """
    status = claim.get("status")
    mapping = {"direct": 0, "reported": 1, "inferred": 1, "disputed": 2}
    if status not in mapping:
        raise ValueError(f"Unmapped claim status for p-adic level a1: {status!r}")
    if claim.get("tag") == "exception":
        return 2
    return mapping[status]


def event_phase(claim: dict[str, Any], document: dict[str, Any] | None = None) -> int:
    """a2: 0=before restriction, 1=during incident, 2=after/unknown."""
    text = f"{claim.get('text', '')} {claim.get('evidence', '')}"
    for match in TIME_RE.finditer(text):
        hour, minute = int(match.group(1)), int(match.group(2))
        second = int(match.group(3) or 0)
        hhmmss = (hour, minute, second)
        if hhmmss < (18, 42, 16):
            return 0
        if hhmmss <= (19, 12, 19):
            return 1
        return 2
    # If the claim has no event time, use the document timestamp only to classify the era.
    # If neither gives a usable time, it belongs to the after/unknown level.
    created = (document or {}).get("created_at", "")
    try:
        dt = datetime.fromisoformat(created)
        incident = datetime.fromisoformat(INCIDENT_RESTRICTION)
        restored = datetime.fromisoformat(INCIDENT_RESTORED)
        if dt < incident:
            return 0
        if dt <= restored:
            return 1
        return 2
    except (TypeError, ValueError):
        return 2


def p_adic_path(claim: dict[str, Any], document: dict[str, Any]) -> tuple[int, int, int]:
    """Return [a0,a1,a2] in the precise field order defined by the assignment."""
    return (source_level(document), status_level(claim), event_phase(claim, document))


def p_adic_code(path: tuple[int, int, int] | list[int], p: int = 3) -> int:
    if p != 3:
        raise ValueError("This lab specifies p=3.")
    if len(path) != 3 or any(not 0 <= int(a) < p for a in path):
        raise ValueError(f"Expected a length-3 path with digits in [0,2], got {path!r}")
    a0, a1, a2 = (int(x) for x in path)
    return a0 + a1 * p + a2 * (p ** 2)


def p_adic_distance(x: tuple[int, ...] | list[int], y: tuple[int, ...] | list[int], p: int = 3) -> float:
    """Lab-defined branch distance: 0 for same path; otherwise 3**(-k), k=shared prefix length."""
    if p != 3:
        raise ValueError("This lab specifies p=3.")
    if len(x) != 3 or len(y) != 3:
        raise ValueError("p-adic paths must contain exactly three levels")
    if tuple(x) == tuple(y):
        return 0.0
    k = 0
    for a, b in zip(x, y):
        if a != b:
            break
        k += 1
    return p ** (-k)
