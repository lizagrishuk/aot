import sys, json, os
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from schemas import parse_model_answer, ModelAnswer
from llm import generate_raw_answer
VALID={"C01","C02","C03"}

def response(**updates):
    obj={"question_id":"Q01","status":"entailed","answer":"18:42:16","evidence_ids":["C01"],"counterevidence_ids":[],"confidence":0.9,"explanation":"Время прямо указано в доступном источнике."}
    obj.update(updates)
    return json.dumps(obj,ensure_ascii=False)

def test_invalid_json_rejected():
    with pytest.raises(ValueError,match="valid JSON"):
        parse_model_answer("not json",VALID,"Q01")

def test_nonexistent_evidence_id_rejected():
    with pytest.raises(ValueError,match="unknown evidence IDs"):
        parse_model_answer(response(evidence_ids=["C999"]),VALID,"Q01")

def test_both_without_counterevidence_rejected():
    with pytest.raises(ValueError,match="requires non-empty"):
        parse_model_answer(response(status="both",evidence_ids=["C01"],counterevidence_ids=[]),VALID,"Q01")

def test_insufficient_with_invented_fact_rejected():
    bad=response(status="insufficient",answer="Артём точно использовал скрипт.",explanation="Недостаточно данных, чтобы установить, использовал ли он скрипт.",evidence_ids=[])
    with pytest.raises(ValueError,match="must not assert a specific fact"):
        parse_model_answer(bad,VALID,"Q01")

def test_valid_insufficient_must_state_missing_information():
    good=response(status="insufficient",answer="Недостаточно данных, чтобы установить использование скрипта.",explanation="Не хватает независимого технического журнала ввода.",evidence_ids=[],counterevidence_ids=[],confidence=0.3)
    answer=parse_model_answer(good,VALID,"Q01")
    assert answer.status.value=="insufficient"

def test_confidence_outside_range_rejected():
    with pytest.raises(Exception):
        parse_model_answer(response(confidence=1.2),VALID,"Q01")

def test_unavailable_local_model_is_reported_without_fake_answer(monkeypatch):
    import llm
    monkeypatch.setenv("OLLAMA_URL", "http://127.0.0.1:1")
    with pytest.raises(RuntimeError, match="Ollama недоступна"):
        generate_raw_answer("Q01", "question", "context", api_key=None, valid_claim_ids={"C01"})

from schemas import parse_model_answer_with_safe_repair

def test_safe_repair_downgrades_both_without_counterevidence():
    bad = response(status="both", answer="18:42:16", evidence_ids=["C01"], counterevidence_ids=[])
    answer, reasons, safe = parse_model_answer_with_safe_repair(bad, VALID, "Q01")
    assert answer.status.value == "insufficient"
    assert "недостаточно" in answer.answer.lower()
    assert safe is True
    assert any("both_without_two_sides" in x for x in reasons)

def test_safe_repair_downgrades_insufficient_with_assertive_answer():
    bad = response(status="insufficient", answer="Аккаунт ограничили в 18:42:16.",
                   explanation="Событие произошло в 18:42:16.", evidence_ids=["C01"])
    answer, reasons, safe = parse_model_answer_with_safe_repair(bad, VALID, "Q01")
    assert answer.status.value == "insufficient"
    assert "недостаточно" in answer.answer.lower()
    assert safe is True

def test_safe_repair_filters_unknown_claim_ids_without_fabricating_new_ids():
    bad = response(status="entailed", answer="18:42:16", evidence_ids=["C999", "C01"])
    answer, reasons, safe = parse_model_answer_with_safe_repair(bad, VALID, "Q01")
    assert answer.evidence_ids == ["C01"]
    assert "C999" not in answer.evidence_ids
    assert "evidence_ids_normalized_or_filtered" in reasons
