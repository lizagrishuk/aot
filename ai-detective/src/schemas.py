"""Pydantic validation for the Lab 6 JSON contract, with a transparent fail-safe wrapper.

`parse_model_answer` is intentionally strict and is used by tests. If a small local
model still returns a semantically inconsistent but parseable JSON after one repair
attempt, `parse_model_answer_with_safe_repair` produces a conservative abstention and
returns repair metadata. The raw model output must always be retained by the caller.
"""
from __future__ import annotations
import json
import re
from difflib import get_close_matches
from enum import Enum
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, model_validator

from common import ANSWER_MAX_CHARS, EXPLANATION_MAX_CHARS


class AnswerStatus(str, Enum):
    entailed = "entailed"
    contradicted = "contradicted"
    both = "both"
    insufficient = "insufficient"

class ModelAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question_id: str = Field(min_length=1)
    status: AnswerStatus
    answer: str | None
    evidence_ids: list[str] = Field(default_factory=list)
    counterevidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str = Field(min_length=1)

    @model_validator(mode="after")
    def logical_requirements(self):
        if self.status == AnswerStatus.both and (not self.evidence_ids or not self.counterevidence_ids):
            raise ValueError("status='both' requires non-empty evidence_ids and counterevidence_ids")
        if self.status == AnswerStatus.insufficient:
            markers = ("недостаточно", "не хватает", "не позволяет установить", "нет данных", "неизвестно",
                       "cannot determine", "insufficient", "нельзя установить", "нельзя подтвердить")
            answer_text = (self.answer or "").lower()
            if not any(m in answer_text for m in markers):
                raise ValueError("status='insufficient' must not assert a specific fact as the answer; answer must explicitly state uncertainty")
        return self


def parse_model_answer(raw_text: str, valid_claim_ids: set[str], expected_question_id: str | None = None) -> ModelAnswer:
    """Strict parsing; invalid JSON, IDs, and inconsistent statuses are rejected."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ValueError("LLM returned an empty response")
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM response is not valid JSON: {exc.msg}") from exc
    answer = ModelAnswer.model_validate(payload)
    if expected_question_id is not None and answer.question_id != expected_question_id:
        raise ValueError(f"Question ID mismatch: expected {expected_question_id}, got {answer.question_id}")
    unknown = (set(answer.evidence_ids) | set(answer.counterevidence_ids)) - valid_claim_ids
    if unknown:
        raise ValueError(f"Model referenced unknown evidence IDs: {sorted(unknown)}")
    return answer


UNCERTAINTY_MARKERS = (
    "недостаточно", "не хватает", "не позволяет установить", "нет данных", "неизвестно",
    "cannot determine", "insufficient", "нельзя установить", "нельзя подтвердить"
)


def _safe_abstention(question_id: str, confidence: float = 0.1) -> ModelAnswer:
    return ModelAnswer(
        question_id=question_id,
        status=AnswerStatus.insufficient,
        answer="Недостаточно данных: ответ модели не прошёл логическую проверку, поэтому вывод не принимается автоматически.",
        evidence_ids=[],
        counterevidence_ids=[],
        confidence=max(0.0, min(float(confidence), 0.2)),
        explanation="Недостаточно данных: ответ и его доказательства были несогласованы; требуется ручная проверка контекста.",
    )


def _list_of_ids(value: Any, allowed: set[str]) -> list[str]:
    if isinstance(value, str):
        candidates = re.findall(r"\bC\d{2}\b|\bCX\d+\b", value)
    elif isinstance(value, (list, tuple, set)):
        candidates = []
        for item in value:
            if isinstance(item, str):
                candidates.extend(re.findall(r"\bC\d{2}\b|\bCX\d+\b", item))
    else:
        candidates = []
    return list(dict.fromkeys(x for x in candidates if x in allowed))[:4]


def parse_model_answer_with_safe_repair(
    raw_text: str,
    valid_claim_ids: set[str],
    expected_question_id: str,
) -> tuple[ModelAnswer, list[str], bool]:
    """Last-resort schema repair after a failed strict parse and a failed model retry.

    Returns (answer, repair_reasons, safe_abstention_used). This does NOT hide native
    model failures: callers must store the raw output and mark validation_status as
    `safe_fallback` whenever the answer was replaced by a conservative abstention.
    """
    try:
        return parse_model_answer(raw_text, valid_claim_ids, expected_question_id), [], False
    except Exception as strict_error:
        reasons = [f"strict_validation_failed: {type(strict_error).__name__}: {strict_error}"]

    try:
        payload = json.loads(raw_text)
    except Exception as exc:
        reasons.append(f"invalid_json_after_retry: {type(exc).__name__}")
        return _safe_abstention(expected_question_id), reasons, True
    if not isinstance(payload, dict):
        reasons.append("json_root_not_object")
        return _safe_abstention(expected_question_id), reasons, True

    # Some small models put answer metadata inside `answer`; unwrap known fields safely.
    answer_value = payload.get("answer")
    if isinstance(answer_value, dict):
        nested = answer_value
        for key in ("evidence_ids", "counterevidence_ids", "confidence", "explanation"):
            if key not in payload and key in nested:
                payload[key] = nested[key]
        answer_value = nested.get("text", nested.get("value"))
        reasons.append("nested_answer_object_unwrapped")

    model_qid = payload.get("question_id")
    if model_qid != expected_question_id:
        payload["question_id"] = expected_question_id
        reasons.append("question_id_normalized")

    raw_status = str(payload.get("status", "")).strip().lower()
    status_aliases = {
        "entailed": "entailed", "entailment": "entailed", "entailed_status": "entailed", "entalles": "entailed", "entalled": "entailed",
        "contradicted": "contradicted", "contradiction": "contradicted",
        "both": "both", "conflict": "both", "disputed": "both",
        "insufficient": "insufficient", "unknown": "insufficient", "not_enough_information": "insufficient",
    }
    status = status_aliases.get(raw_status)
    if status is None:
        close = get_close_matches(raw_status, [x.value for x in AnswerStatus], n=1, cutoff=0.72)
        status = close[0] if close else "insufficient"
        reasons.append(f"status_normalized:{raw_status or '<empty>'}->{status}")
    elif status != raw_status:
        reasons.append(f"status_alias_normalized:{raw_status}->{status}")

    evidence_ids = _list_of_ids(payload.get("evidence_ids", []), valid_claim_ids)
    counter_ids = _list_of_ids(payload.get("counterevidence_ids", []), valid_claim_ids)
    if evidence_ids != payload.get("evidence_ids", []):
        reasons.append("evidence_ids_normalized_or_filtered")
    if counter_ids != payload.get("counterevidence_ids", []):
        reasons.append("counterevidence_ids_normalized_or_filtered")

    try:
        confidence = float(payload.get("confidence", 0.1))
    except Exception:
        confidence = 0.1
        reasons.append("confidence_defaulted")
    if not (0.0 <= confidence <= 1.0):
        confidence = max(0.0, min(confidence, 1.0))
        reasons.append("confidence_clamped")

    if not isinstance(answer_value, str):
        answer_value = ""
    answer_value = answer_value.strip()[:ANSWER_MAX_CHARS]
    explanation = payload.get("explanation")
    if not isinstance(explanation, str) or not explanation.strip():
        explanation = "Модель не предоставила проверяемого объяснения."
        reasons.append("explanation_defaulted")
    explanation = explanation.strip()[:EXPLANATION_MAX_CHARS]

    # If status and evidence contradict each other, do not invent citations. Abstain safely.
    logical_problem = None
    if status == "both" and (not evidence_ids or not counter_ids):
        logical_problem = "both_without_two_sides"
    elif status in {"entailed", "contradicted"} and not (evidence_ids or counter_ids):
        logical_problem = "assertive_status_without_evidence"
    elif status == "insufficient" and not any(m in answer_value.lower() for m in UNCERTAINTY_MARKERS):
        logical_problem = "insufficient_with_assertive_answer"
    elif not answer_value:
        logical_problem = "missing_answer_text"

    if logical_problem:
        reasons.append(f"safe_abstention:{logical_problem}")
        # Retain only citations that survived the allowed-ID filter, but never fabricate IDs.
        safe = _safe_abstention(expected_question_id, confidence)
        if evidence_ids or counter_ids:
            safe = safe.model_copy(update={"evidence_ids": evidence_ids, "counterevidence_ids": counter_ids})
        return safe, reasons, True

    candidate = {
        "question_id": expected_question_id,
        "status": status,
        "answer": answer_value,
        "evidence_ids": evidence_ids,
        "counterevidence_ids": counter_ids,
        "confidence": confidence,
        "explanation": explanation,
    }
    try:
        answer = ModelAnswer.model_validate(candidate)
        return answer, reasons, False
    except Exception as exc:
        reasons.append(f"safe_abstention:normalization_still_invalid:{type(exc).__name__}")
        return _safe_abstention(expected_question_id, confidence), reasons, True
