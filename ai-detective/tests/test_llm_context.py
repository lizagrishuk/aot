import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import load_documents, load_claims
from context_builder import full_context

def test_full_context_keeps_all_documents_and_compact_claim_registry():
    docs, claims = load_documents(), load_claims()
    context = full_context(docs, claims)
    for d in docs:
        assert f'[{d["document_id"]}] {d["title"]}' in context
        assert d["text"] in context
    assert "КОМПАКТНЫЙ РЕЕСТР ID УТВЕРЖДЕНИЙ" in context
    # The document text itself is retained, while duplicate claim evidence is not repeated in the index.
    assert all(f'[{c["claim_id"]}; {c["document_id"]}; {c["status"]}] {c["text"]}' in context for c in claims)
