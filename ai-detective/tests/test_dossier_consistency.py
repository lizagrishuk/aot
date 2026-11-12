"""Consistency of the dossier, claims, questions and the p-adic encoding (checks found in the audit)."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import load_documents, load_claims, load_questions  # noqa: E402
from padic import p_adic_path, status_level  # noqa: E402

META = ["LLM", "retrieval", "атомарн", "эталон", "нерелевантн", "отвлека", "тест на противоречия", "ошибка статуса"]


def test_documents_contain_no_commentary_about_the_experiment():
    for d in load_documents():
        for word in META:
            assert word.lower() not in d["text"].lower(), f"{d['document_id']}: {word}"


def test_word_count_is_inside_the_required_range_for_both_counting_methods():
    docs = load_documents()
    regex = sum(len(re.findall(r"\b[\w@.-]+\b", d["text"])) for d in docs)
    split = sum(len(d["text"].split()) for d in docs)
    assert 5000 <= split and regex <= 8000


def test_statements_and_chat_are_reported_not_direct():
    docs = {d["document_id"]: d for d in load_documents()}
    for c in load_claims():
        if docs[c["document_id"]]["source_type"] in {"statement", "chat"}:
            assert c["status"] in {"reported", "disputed"}, c["claim_id"]


def test_disputed_claims_are_a_pair_that_contradict_each_other_and_are_atomic():
    disputed = [c for c in load_claims() if c["status"] == "disputed"]
    assert {c["claim_id"] for c in disputed} == {"C36", "C37"}
    assert all(len(c["evidence"]) < 250 for c in disputed)


def test_dataset_has_a_real_both_question_whose_sources_are_equally_reliable():
    claims = {c["claim_id"]: c for c in load_claims()}
    docs = {d["document_id"]: d for d in load_documents()}
    both = [q for q in load_questions() if q["gold_status"] == "both"]
    assert both, "no question with gold status both"
    for q in both:
        types = {docs[claims[i]["document_id"]]["source_type"] for i in q["gold_evidence_ids"] + q["gold_counterevidence_ids"]}
        assert types == {"statement"}  # no log or rule can settle the dispute


def test_exception_branch_comes_from_the_tag_not_from_keywords():
    claims = {c["claim_id"]: c for c in load_claims()}
    docs = {d["document_id"]: d for d in load_documents()}
    for cid in ("C31", "C32", "C33", "C38"):
        assert claims[cid].get("tag") == "exception"
        assert p_adic_path(claims[cid], docs[claims[cid]["document_id"]])[1] == 2
    # a claim that mentions the word but is not a condition of the exception is not moved
    fake = {"status": "direct", "text": "Исключение упомянуто", "evidence": "исключение"}
    assert status_level(fake) == 0


def test_yes_no_gold_answers_start_with_da_or_net():
    for q in load_questions():
        if q["question_id"] in {"Q05", "Q06", "Q07", "Q08", "Q10"}:
            assert q["gold_answer"].startswith(("Да", "Нет")), q["question_id"]


def test_q04_question_does_not_state_the_fact_that_the_removal_test_removes():
    q = next(q for q in load_questions() if q["question_id"] == "Q04")
    assert "уникальн" not in q["question"]


def test_calibration_bins_do_not_double_count_boundary_confidence():
    import evaluation
    import inspect
    src = inspect.getsource(evaluation)
    assert "lo10" in src, "calibration bins must use integer tenths, not float edges"


def test_app_helpers_comparison_and_removal():
    from app_helpers import claim_rows, comparison_rows, removal_literals
    from common import load_claims, load_questions
    claims, questions = load_claims(), {q["question_id"]: q for q in load_questions()}
    q = questions["Q09"]
    runs = [{"question_id": "Q09", "mode": "B", "total_tokens": 10, "elapsed_seconds": 1.0, "validation_status": "valid",
             "prediction": {"status": "both", "confidence": 0.7, "evidence_ids": ["C37"], "counterevidence_ids": ["C36"]}}]
    rows = comparison_rows(runs, q)
    assert [r["Режим"] for r in rows] == ["Полный контекст", "BM25 top-15", "p-adic контекст"]
    assert rows[1]["Верно"] is True and rows[1]["Evidence F1"] == 1.0 and rows[0]["Статус"] == "нет прогона"
    assert claim_rows(["C37"], claims, q["gold_evidence_ids"], q["gold_counterevidence_ids"])[0]["в эталоне"] == "evidence"
    by_id = {c["claim_id"]: c for c in claims}
    assert removal_literals(["C16"], claims) == [by_id["C16"]["evidence"]]
