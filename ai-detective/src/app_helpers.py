"""Pure helpers for the Streamlit stand (no Streamlit import, so they are unit-tested)."""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any

from evaluation import _prf

MODE_NAMES = {"A": "Полный контекст", "B": "BM25 top-15", "C": "p-adic контекст"}


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def context_claim_ids(context: str) -> list[str]:
    """Return only claim IDs explicitly present in the post-intervention model context."""
    return sorted(set(re.findall(r"^\[(C\d{2}|CX\d+);", context, flags=re.MULTILINE)))


def context_document_ids(context: str, fallback: list[str] | None = None) -> list[str]:
    """Extract visible full-document order (mode A), or use retrieval source IDs for B/C."""
    visible = re.findall(r"^\[(D\d{2})\]", context, flags=re.MULTILINE)
    return visible if visible else list(fallback or [])


def claim_rows(ids: list[str], claims: list[dict], gold_evidence: list[str], gold_counter: list[str]) -> list[dict]:
    """Claim ids with their text, source and whether the gold standard lists them."""
    by_id = {c["claim_id"]: c for c in claims}
    rows = []
    for cid in ids:
        c = by_id.get(cid, {})
        rows.append({"claim_id": cid, "document": c.get("document_id", "?"), "status": c.get("status", "?"),
                     "text": c.get("text", "(нет в досье)"),
                     "в эталоне": "evidence" if cid in gold_evidence else ("counter" if cid in gold_counter else "—")})
    return rows


def removal_literals(selected_ids: list[str], claims: list[dict]) -> list[str]:
    """Verbatim source fragments of the chosen claims: removing them removes the whole fact."""
    by_id = {c["claim_id"]: c for c in claims}
    return [by_id[i]["evidence"] for i in selected_ids if i in by_id]


def comparison_rows(runs: list[dict], question: dict) -> list[dict]:
    """One row per mode for the selected question, like the example screen of the assignment."""
    gold_e, gold_c = set(question["gold_evidence_ids"]), set(question["gold_counterevidence_ids"])
    latest: dict[str, dict] = {}
    for r in runs:
        if r.get("question_id") == question["question_id"] and isinstance(r.get("prediction"), dict):
            latest[r.get("mode")] = r
    rows = []
    for mode in ("A", "B", "C"):
        r = latest.get(mode)
        if r is None:
            rows.append({"Режим": MODE_NAMES[mode], "Статус": "нет прогона"})
            continue
        p = r["prediction"]
        ev = _prf(set(p.get("evidence_ids", [])), gold_e)
        co = _prf(set(p.get("counterevidence_ids", [])), gold_c)
        rows.append({"Режим": MODE_NAMES[mode], "Статус": p.get("status"),
                     "Верно": p.get("status") == question["gold_status"],
                     "Confidence": p.get("confidence"),
                     "Evidence F1": ev["f1"], "Counter recall": co["recall"] if gold_c else None,
                     "Токены": r.get("total_tokens"), "Время, с": r.get("elapsed_seconds"),
                     "Валидация": r.get("validation_status"), "Источник": r.get("source", "эксперимент")})
    return rows
