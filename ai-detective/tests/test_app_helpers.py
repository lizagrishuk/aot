import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from app_helpers import context_claim_ids, context_document_ids


def test_context_claim_ids_only_returns_ids_present_after_intervention():
    context = """[D01] Document one
[C01; status=direct] Fact one
[CX01; status=direct] Added synthetic fact
"""
    assert context_claim_ids(context) == ["C01", "CX01"]


def test_context_document_ids_preserves_full_document_order_and_retrieval_fallback():
    full = "[D03] Third\nbody\n[D01] First\nbody\n"
    assert context_document_ids(full, ["D99"]) == ["D03", "D01"]
    assert context_document_ids("[C01; status=direct] Fact", ["D02", "D01"]) == ["D02", "D01"]
