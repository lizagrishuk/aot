"""Regression tests for the 12 manually reviewed gold records and Q06 removal.

These are *structural* checks, not substitutes for independent manual sign-off.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import load_documents, load_claims, load_questions  # noqa: E402
from context_builder import build_contexts  # noqa: E402
from interventions import remove_fact  # noqa: E402
from stress_tests import REMOVALS  # noqa: E402


def test_all_gold_references_are_real_exact_quotes_and_document_linked():
    docs = {d["document_id"]: d for d in load_documents()}
    claims = {c["claim_id"]: c for c in load_claims()}
    for q in load_questions():
        refs = q["gold_evidence_ids"] + q["gold_counterevidence_ids"]
        assert len(refs) == len(set(refs)), q["question_id"]
        assert refs, q["question_id"]
        for cid in refs:
            c = claims[cid]
            assert c["evidence"] in docs[c["document_id"]]["text"], (q["question_id"], cid)
        if q["gold_status"] == "both":
            assert q["gold_evidence_ids"] and q["gold_counterevidence_ids"]


def test_gold_q07_and_q11_refer_to_the_precise_primary_facts():
    q = {x["question_id"]: x for x in load_questions()}
    c = {x["claim_id"]: x for x in load_claims()}
    assert "C02" in q["Q07"]["gold_evidence_ids"]
    assert c["C02"]["document_id"] == "D01"  # direct incident-card record
    assert "C26" in q["Q11"]["gold_evidence_ids"]
    assert "unique_message_count=1" in c["C26"]["evidence"]
    assert "C27" not in q["Q11"]["gold_evidence_ids"]  # restriction reason != unique-message proof


def test_q09_is_a_real_two_sided_witness_disagreement():
    docs = {d["document_id"]: d for d in load_documents()}
    claims = {c["claim_id"]: c for c in load_claims()}
    q = {x["question_id"]: x for x in load_questions()}["Q09"]
    assert q["gold_status"] == "both"
    assert q["gold_evidence_ids"] == ["C37"]
    assert q["gold_counterevidence_ids"] == ["C36"]
    assert claims["C36"]["status"] == claims["C37"]["status"] == "disputed"
    assert "не появлялось" in claims["C36"]["evidence"]
    assert "было показано" in claims["C37"]["evidence"]
    assert docs["D03"]["source_type"] == docs["D05"]["source_type"] == "statement"
    assert "в отдельном журнале пользовательских уведомлений" in docs["D07"]["text"]


def test_q06_removal_is_necessary_and_cannot_be_reconstructed_from_other_sources():
    q = next(x for x in load_questions() if x["question_id"] == "Q06")
    docs, claims = load_documents(), load_claims()
    contexts = build_contexts(q["question"], 15, docs, claims)
    cfg = REMOVALS["Q06"]
    assert cfg["expected"] == "insufficient"
    for mode in ("A", "B", "C"):
        before = contexts[mode]["context"]
        after, dropped, n = remove_fact(before, cfg["literals"], mode, claims)
        assert after != before
        assert "C29" in dropped, mode
        assert "C17" in dropped, mode
        assert "[C28;" in after, mode  # same 19:10 approval, different operation
        assert "19:10:06" in after, mode
        assert "19:12" not in after, mode
        assert "— account_unrestricted user=usr_4821" not in after, mode
        assert not any(x in after for x in cfg["literals"]), mode
        if mode == "A":
            assert n >= 7
