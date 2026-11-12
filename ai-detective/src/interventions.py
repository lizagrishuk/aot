"""Context interventions for stress test 2 (removal of one fact).

One *fact* is removed, not one text line. A fact in a dossier is usually stated in several places
(a log line, the timeline, a witness statement), so every sentence that states it is masked and every
claim that states it is dropped. The fact is identified by a few literal markers; after the
intervention none of them may remain in the context (checked in build_cases and in tests).
"""
from __future__ import annotations
import re

MARKER = "[СОДЕРЖАНИЕ УДАЛЕНО В СТРЕСС-ТЕСТЕ]"
REGISTRY_MARKER = "\n\nКОМПАКТНЫЙ РЕЕСТР ID УТВЕРЖДЕНИЙ"
# A sentence ends at ".", "!", "?" or "»." followed by spaces, or at a line break (log records have no punctuation).
SPLIT_RE = re.compile(r"(\n+|(?<=[.!?»])[ \t]+)")
CLAIM_LINE_RE = re.compile(r'^\[(C\d{2}|CX\d+); source=([^;\]]+); status=([^\]]+)\] (.*?) Evidence: "(.*)"$')


def _has(text: str, literals: list[str]) -> bool:
    return any(lit in text for lit in literals)


def mask_text(text: str, literals: list[str]) -> tuple[str, int]:
    """Replace every sentence/record containing a literal with MARKER. Returns (text, number of masked pieces)."""
    parts = SPLIT_RE.split(text)
    masked = 0
    for i in range(0, len(parts), 2):  # even indexes are sentences, odd ones are separators
        if parts[i] and _has(parts[i], literals):
            parts[i] = MARKER
            masked += 1
    return "".join(parts), masked


def _remove_from_claim_line(line: str, literals: list[str]) -> tuple[str | None, bool]:
    """Return (new_line or None if the claim is dropped, changed)."""
    m = CLAIM_LINE_RE.match(line)
    if not m:
        return line, False
    cid, src, status, text, evidence = m.groups()
    if _has(text, literals):
        return None, True
    pieces = [p for p in SPLIT_RE.split(evidence)[0::2] if p]
    if pieces and all(_has(p, literals) for p in pieces):
        return None, True
    new_evidence, n = mask_text(evidence, literals)
    if n == 0:
        return line, False
    return f'[{cid}; source={src}; status={status}] {text} Evidence: "{new_evidence}"', True


def claim_states_fact(claim: dict, literals: list[str]) -> bool:
    """A claim is dropped if its text states the fact or its whole evidence quote would be masked."""
    if _has(claim["text"], literals):
        return True
    pieces = [p for p in SPLIT_RE.split(claim["evidence"])[0::2] if p]
    return bool(pieces) and all(_has(p, literals) for p in pieces)


def remove_fact(context: str, literals: list[str], mode: str, claims: list[dict] | None = None) -> tuple[str, list[str], int]:
    """Remove one fact from an A/B/C context. Returns (mutated context, dropped claim ids, masked source pieces).

    Mode A needs the claim list: its registry shows the claim text only, so the decision to drop a
    claim has to use the evidence quote as well.
    """
    dropped: list[str] = []
    masked = 0
    if mode == "A":
        if claims is None:
            raise ValueError("mode A needs the claims list")
        drop_ids = {c["claim_id"] for c in claims if claim_states_fact(c, literals)}
        head, sep, tail = context.partition(REGISTRY_MARKER)
        head, masked = mask_text(head, literals)
        lines = []
        for line in tail.splitlines():
            m = re.match(r"^\[(C\d{2}|CX\d+); [^;\]]+; [^\]]+\] (.*)$", line)
            if m and m.group(1) in drop_ids:
                dropped.append(m.group(1))
                continue
            lines.append(line)
        return head + sep + "\n".join(lines), dropped, masked
    out: list[str] = []
    for line in context.splitlines():
        new, changed = _remove_from_claim_line(line, literals)
        if new is None:
            dropped.append(CLAIM_LINE_RE.match(line).group(1))
            continue
        if changed:
            masked += 1
        out.append(new)
    if mode == "C":  # drop branch headers left without claims
        cleaned: list[str] = []
        for i, line in enumerate(out):
            if line.startswith("ВЕТВЬ ") and not (i + 1 < len(out) and out[i + 1].startswith("[")):
                continue
            cleaned.append(line)
        out = cleaned
    return "\n".join(out), dropped, masked
