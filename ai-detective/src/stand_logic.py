"""Deterministic, UI-independent operations of the AI-detective research stand.

The model experiment, dossier and recorded results stay untouched.  This module
applies the *original* interventions to the exact model input from the experiment
(or to a freshly built pack for an uploaded dossier), and matches archived model
answers only when their input SHA-256 is byte-for-byte identical.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from app_helpers import context_claim_ids
from context_builder import build_contexts, full_context
from interventions import SPLIT_RE, remove_fact
from stress_tests import (
    CONTRADICTIONS, REMOVALS, _add_synthetic, permute_source_documents,
    permuted_document_context,
)

STRESS_NAMES = {
    "none": "Без вмешательства",
    "permutation": "Перестановка документов (seed 42)",
    "remove_premise": "Удаление выбранной посылки",
    "add_contradiction": "Добавление подготовленного противоречия",
}


def sha256_context(context: str) -> str:
    return hashlib.sha256(context.encode("utf-8")).hexdigest()


def jsonl_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(row) for row in path.read_text(encoding="utf-8").splitlines() if row.strip()]


def load_archives(root: Path) -> dict[str, Any]:
    """Read-only indexes; caller may cache this function at the Streamlit layer."""
    results = root / "results"
    base_contexts = jsonl_rows(results / "base_contexts.jsonl")
    base_runs = jsonl_rows(results / "llm_runs.jsonl")
    stress_cases = jsonl_rows(results / "stress_cases.jsonl")
    stress_runs = jsonl_rows(results / "stress_runs.jsonl")
    def json_or_none(name: str):
        path = results / name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return {
        "contexts": {(row["question_id"], row["mode"]): row for row in base_contexts},
        "base_runs": {(row["question_id"], row["mode"]): row for row in base_runs},
        "stress_cases": {row["case_id"]: row for row in stress_cases},
        "stress_runs": {row["case_id"]: row for row in stress_runs},
        "base_list": base_runs,
        "stress_list": stress_runs,
        "metrics": json_or_none("llm_metrics.json"),
        "stress_metrics": json_or_none("stress_metrics.json"),
    }


def base_packs(question: dict, documents: list[dict], claims: list[dict],
               archive: dict | None = None, use_recorded_context: bool = True) -> dict[str, dict]:
    """Use original experiment inputs for its fixed dossier, fresh BM25 for uploads.

    Recorded mode B and C context is the *actual* BM25/p-adic model input, and
    avoids an accidental drift across future NLTK/rank-bm25 package versions.
    Dynamic uploads always execute the original `build_contexts()` algorithm.
    """
    qid = question["question_id"]
    if use_recorded_context and archive and all((qid, m) in archive["contexts"] for m in "ABC"):
        result = {}
        for mode in "ABC":
            rec = archive["contexts"][(qid, mode)]
            value = rec["context_text"]
            if sha256_context(value) != rec["context_sha256"]:
                raise ValueError(f"Повреждён сохранённый контекст {qid}/{mode}")
            result[mode] = {
                "context": value,
                "claim_ids": rec["claim_ids"],
                "document_ids": rec["document_ids"],
                "source": "exact_experiment_context",
            }
        if set(result["B"]["claim_ids"]) != set(result["C"]["claim_ids"]):
            raise ValueError(f"B/C: не совпадает набор утверждений для {qid}")
        return result
    result = build_contexts(question["question"], 15, documents, claims)
    for item in result.values():
        item["source"] = "live_context_builder"
    return result


def apply_intervention(
    question: dict, mode: str, base_context: str, documents: list[dict], claims: list[dict],
    kind: str = "none", *, prepared_removal: bool = True,
    selected_claim_id: str | None = None, custom_literals: str = "",
) -> dict[str, Any]:
    """Apply real experiment interventions, not a lookup/substitution by case ID."""
    if mode not in "ABC" or kind not in STRESS_NAMES:
        raise ValueError("Неизвестный режим или тип вмешательства")
    qid = question["question_id"]
    context = base_context
    info = "Исходный контекст без изменений"
    changed = False
    synthetic = None
    official_case_id = None
    if kind == "permutation":
        shuffled, before, after = permute_source_documents(documents, seed=42)
        # Mode A reads full documents. B/C read the already selected claim set;
        # permuting document input is a control of their retrieval invariance.
        if mode == "A":
            context = full_context(shuffled, claims)
        else:
            # Really rerun retrieval before and after permuting the *source*
            # list. It must be invariant in B/C. Preserve the exact recorded
            # baseline context to avoid mixing different tokenizer versions
            # when comparing against the archived Qwen experiment.
            original_retrieval = permuted_document_context(mode, question, documents, claims)
            permuted_retrieval = permuted_document_context(mode, question, shuffled, claims)
            if original_retrieval != permuted_retrieval:
                raise AssertionError("Перестановка документов изменила B/C retrieval")
            context = base_context
        info = f"Порядок документов: {', '.join(before)} → {', '.join(after)}"
        official_case_id = f"{qid}_{mode}_permutation"
    elif kind == "remove_premise":
        user_markers = [p.strip() for p in custom_literals.split(";") if p.strip()]
        if user_markers:
            literals = user_markers
            removal_source = "пользовательские маркеры"
        elif selected_claim_id:
            selected = next((c for c in claims if c["claim_id"] == selected_claim_id), None)
            if selected is None:
                raise ValueError(f"Утверждение {selected_claim_id} не найдено")
            # Remove the chosen fact's own statement and source quote.
            # A quote may contain several sentences. The source masker works
            # sentence-by-sentence; a full multi-sentence string would never
            # match an individual piece and could leave the fact untouched.
            evidence_parts = [part.strip() for part in SPLIT_RE.split(selected["evidence"])[::2]
                              if len(part.strip()) >= 8]
            literals = list(dict.fromkeys([selected["text"], *evidence_parts]))
            removal_source = f"выбранное утверждение {selected_claim_id}"
        elif prepared_removal and qid in REMOVALS:
            literals = REMOVALS[qid]["literals"]
            removal_source = "подготовленная посылка эксперимента"
            official_case_id = f"{qid}_{mode}_remove_premise"
        else:
            raise ValueError("Выбери утверждение или задай маркеры удаляемого факта")
        context, dropped, masked = remove_fact(base_context, literals, mode, claims)
        leftovers = [marker for marker in literals if marker in context]
        if leftovers:
            raise ValueError("Часть удаляемого факта осталась в контексте: " + "; ".join(leftovers[:2]))
        info = f"Источник: {removal_source}. Удалены ID: {', '.join(dropped) or 'нет'}. Замаскировано фрагментов: {masked}"
    elif kind == "add_contradiction":
        spec = CONTRADICTIONS.get(qid)
        if spec is None:
            raise ValueError(f"Для {qid} не подготовлено противоречие")
        context, synthetic = _add_synthetic(mode, base_context, spec)
        info = f"Добавлен CX01 из источника X01 ({spec['source_type']}); ожидаемый статус: {spec['expected']}"
        official_case_id = f"{qid}_{mode}_add_contradiction"
    changed = context != base_context
    return {
        "context": context,
        "sha256": sha256_context(context),
        "context_changed": changed,
        "claim_ids": context_claim_ids(context),
        "note": info,
        "kind": kind,
        "synthetic_claim": synthetic,
        "official_case_id": official_case_id,
    }


def matched_result(archive: dict[str, Any], question_id: str, mode: str,
                   result: dict[str, Any], is_original_dossier: bool) -> tuple[dict | None, str]:
    """Return an archived answer ONLY when prompt context SHA-256 matches.

    This prevents showing a response to one dossier as if the model answered
    another one (especially after a custom deletion or a document upload).
    """
    if not is_original_dossier:
        return None, "Досье заменено: архив относится только к исходному набору документов"
    key = result["official_case_id"]
    if result["kind"] == "none":
        candidate = archive["base_runs"].get((question_id, mode))
        candidate_sha = (candidate or {}).get("context_sha256")
    else:
        candidate = archive["stress_runs"].get(key) if key else None
        candidate_sha = (candidate or {}).get("mutated_context_sha256")
    if candidate is None:
        return None, "Для этого конкретного вмешательства нет сохранённого ответа; нужен новый запуск Ollama"
    if candidate_sha != result["sha256"]:
        return None, "Контекст отличается от сохранённого эксперимента; старый ответ нельзя выдавать за новый"
    return candidate, "Сохранённый ответ Qwen на точно такой же контекст (SHA-256 совпадает)"
