import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stress_tests import build_cases  # noqa: E402

CLAIM_RE = re.compile(r"^\[(C\d{2}|CX\d+);", re.MULTILINE)


def test_all_stress_cases_make_the_intended_single_intervention():
    cases = build_cases()
    skipped = getattr(build_cases, "skipped_noop", [])
    # 12 questions x 3 modes x (permutation + contradiction) + removal for Q03-Q06 x 3 modes; a removal is skipped only if the fact is not in a B/C context
    assert len(cases) + len(skipped) == 84
    assert len({case["case_id"] for case in cases}) == len(cases)

    for case in cases:
        original = case["original_context"]
        mutated = case["mutated_context"]
        assert hashlib.sha256(original.encode("utf-8")).hexdigest() == case["original_context_sha256"]
        assert hashlib.sha256(mutated.encode("utf-8")).hexdigest() == case["mutated_context_sha256"]

        if case["test_type"] == "remove_premise":
            assert mutated != original, f"No-op premise removal: {case['case_id']}"
            assert case["intervention"]["removed_claim_ids"] or case["intervention"]["masked_source_pieces"]
            # the whole fact is gone, not one line of it: none of its literal markers remains anywhere in the context
            assert not [lit for lit in case["intervention"]["fact_literals"] if lit in mutated], case["case_id"]
            assert set(CLAIM_RE.findall(mutated)) <= set(CLAIM_RE.findall(original))
        elif case["test_type"] == "add_contradiction":
            assert mutated != original
            assert "CX01" in mutated
            assert "NOT A FACT" not in mutated and "SYNTHETIC" not in mutated  # model must not see a test banner
            assert case["expected_stress_status"] in {"both", case["gold_status"]}
            assert set(CLAIM_RE.findall(original)) < set(CLAIM_RE.findall(mutated))
        elif case["test_type"] == "permutation":
            before_ids = CLAIM_RE.findall(original)
            after_ids = CLAIM_RE.findall(mutated)
            assert set(before_ids) == set(after_ids), f"Permutation changed claim membership: {case['case_id']}"
            intervention = case["intervention"]
            assert intervention["permuted_unit"] == "source documents"
            assert intervention["random_state"] == 42
            assert intervention["document_order_before"] != intervention["document_order_after"]
            assert set(intervention["document_order_before"]) == set(intervention["document_order_after"])
            if case["mode"] == "A":
                assert mutated != original, f"Mode A did not expose the document-order change: {case['case_id']}"
                assert case["context_changed"] is True
                actual_doc_order = re.findall(r"^\[(D\d{2})\]", mutated, flags=re.MULTILINE)
                assert actual_doc_order == intervention["document_order_after"]
            else:
                # B/C consume retrieved claims, not the source documents themselves.
                # Rebuilding with shuffled docs must preserve the exact prompt; these are
                # same-context replication controls, not fake claim/branch permutations.
                assert mutated == original, f"Retrieval context should be invariant: {case['case_id']}"
                assert case["context_changed"] is False
                assert intervention["model_context_changed"] is False
                assert intervention["document_order_after"] != intervention["document_order_before"]


def test_document_permutation_is_applied_once_and_propagated_consistently():
    cases = [c for c in build_cases() if c["test_type"] == "permutation"]
    assert len(cases) == 36
    orders = {(tuple(c["intervention"]["document_order_before"]), tuple(c["intervention"]["document_order_after"])) for c in cases}
    assert len(orders) == 1, "All questions/modes must use the same random_state=42 document permutation"
    assert sum(c["mode"] == "A" and c["context_changed"] is True for c in cases) == 12
    assert sum(c["mode"] == "B" and c["context_changed"] is False for c in cases) == 12
    assert sum(c["mode"] == "C" and c["context_changed"] is False for c in cases) == 12


def test_removal_only_for_multihop_and_temporal_and_never_noop():
    cases = [c for c in build_cases() if c["test_type"] == "remove_premise"]
    assert {c["question_type"] for c in cases} <= {"multi_hop", "temporal"}
    assert len(cases) + len(getattr(build_cases, "skipped_noop", [])) == 12


def test_all_twelve_added_contradictions_have_audited_expected_statuses():
    from common import load_questions
    from stress_tests import CONTRADICTIONS

    questions = {q["question_id"]: q for q in load_questions()}
    all_cases = build_cases()
    case_map = {c["case_id"]: c for c in all_cases}
    expected = {
        "Q01": "both", "Q02": "both", "Q03": "entailed", "Q04": "entailed",
        "Q05": "both", "Q06": "contradicted", "Q07": "entailed", "Q08": "contradicted",
        "Q09": "both", "Q10": "contradicted", "Q11": "insufficient", "Q12": "insufficient",
    }
    assert set(CONTRADICTIONS) == set(questions) == set(expected)
    for qid, spec in CONTRADICTIONS.items():
        assert spec["expected"] == expected[qid]
        assert spec["expected"] == "both" or spec["expected"] == questions[qid]["gold_status"]
        assert spec["text"]
        assert spec["source_type"] in {"system_log", "statement"}
        assert spec["text"] not in "\n".join(d["text"] for d in __import__("common").load_documents())
        for mode in ("A", "B", "C"):
            case = case_map[f"{qid}_{mode}_add_contradiction"]
            assert case["expected_stress_status"] == expected[qid]
            assert case["intervention"]["synthetic_claim"]["claim_id"] == "CX01"
            assert case["intervention"]["synthetic_claim"]["status"] == ("direct" if spec["source_type"] == "system_log" else "reported")
            assert spec["text"] in case["mutated_context"]
            assert "CX01" in case["mutated_context"]


def test_q01_q02_added_logs_conflict_on_the_same_single_transition():
    from common import load_claims, load_questions
    from stress_tests import CONTRADICTIONS

    claims = {c["claim_id"]: c for c in load_claims()}
    questions = {q["question_id"]: q for q in load_questions()}
    case_map = {c["case_id"]: c for c in build_cases()}
    for qid, original_id, original_time, added_time, operation in (
        ("Q01", "C27", "18:42:16", "18:44:09", "account_restricted"),
        ("Q02", "C29", "19:12:19", "19:10:31", "account_unrestricted"),
    ):
        spec = CONTRADICTIONS[qid]
        assert questions[qid]["gold_status"] == "entailed"
        assert claims[original_id]["status"] == "direct"
        assert original_time in claims[original_id]["evidence"]
        assert spec["source_type"] == "system_log" and spec["expected"] == "both"
        assert operation in spec["text"]
        assert added_time in spec["text"]
        assert "the_same_single_" in spec["text"]
        assert "no_second_" in spec["text"]

    for qid in ("Q01", "Q02"):
        for mode in ("A", "B", "C"):
            case = case_map[f"{qid}_{mode}_add_contradiction"]
            assert case["expected_stress_status"] == "both"
            assert case["intervention"]["synthetic_claim"]["status"] == "direct"
            assert "CX01" in case["mutated_context"]


def test_q05_added_log_explicitly_negates_the_pre_restriction_delay():
    """A second delay after restriction is not contradictory: Q05 needs an explicit negation."""
    from common import load_claims, load_questions
    from stress_tests import CONTRADICTIONS

    claims = {c["claim_id"]: c for c in load_claims()}
    questions = {q["question_id"]: q for q in load_questions()}
    case_map = {c["case_id"]: c for c in build_cases()}
    original = claims["C24"]["evidence"]
    restriction = claims["C27"]["evidence"]
    add = CONTRADICTIONS["Q05"]

    assert "2026-03-14T18:40:03" in original
    assert "2026-03-14T18:42:16" in restriction
    assert questions["Q05"]["gold_status"] == "entailed"
    assert add["source_type"] == "system_log" and add["expected"] == "both"
    assert "delay_before_restriction=false" in add["text"]
    assert "до ограничения аккаунта в 18:42:16 задержки очереди не было" in add["text"]
    assert "first_delay_detected_at=2026-03-14T18:43:20" in add["text"]

    for mode in ("A", "B", "C"):
        case = case_map[f"Q05_{mode}_add_contradiction"]
        assert case["expected_stress_status"] == "both"
        assert case["intervention"]["synthetic_claim"]["status"] == "direct"
        assert add["text"] in case["mutated_context"]
        assert "CX01" not in case["original_context"]
        assert "CX01" in case["mutated_context"]
