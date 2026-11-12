"""Build A/B/C contexts. B and C deliberately share the exact same claim IDs."""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime
from typing import Any
from common import load_claims, load_documents
from retrieval import bm25_rank
from padic import p_adic_path, p_adic_code


def enrich_claims(claims: list[dict[str, Any]] | None = None, documents: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    claims = claims if claims is not None else load_claims()
    documents = documents if documents is not None else load_documents()
    docs = {d["document_id"]: d for d in documents}
    enriched = []
    for c in claims:
        doc = docs.get(c["document_id"])
        if doc is None:
            raise ValueError(f"Claim {c.get('claim_id')} references missing document {c.get('document_id')}")
        path = p_adic_path(c, doc)
        enriched.append({**c, "p_adic_path": list(path), "p_adic_code": p_adic_code(path)})
    return enriched


def retrieve_claims(question: str, claims: list[dict[str, Any]] | None = None,
                    documents: list[dict[str, Any]] | None = None, top_k: int = 15) -> list[dict[str, Any]]:
    enriched = enrich_claims(claims, documents)
    searchable = [{**c, "text": c["text"] + " " + c["evidence"]} for c in enriched]
    ranked = bm25_rank(question, searchable, top_k=top_k)
    # BM25 indexes text+evidence, but the model must see the original claim text (no duplicated evidence).
    original = {c["claim_id"]: c for c in enriched}
    return [{**original[r["claim_id"]], "_score": r["_score"]} for r in ranked]


def _event_sort_key(claim: dict[str, Any]) -> tuple[int, str, str]:
    text = claim.get("evidence", "") + " " + claim.get("text", "")
    # ISO timestamp in evidence is preferred; otherwise use first HH:MM(:SS), otherwise last.
    import re
    iso = re.search(r"2026-\d\d-\d\dT\d\d:\d\d:\d\d", text)
    if iso:
        return (0, iso.group(0), claim["claim_id"])
    hm = re.search(r"(?<!\d)(\d{1,2}:\d{2}(?::\d{2})?)(?!\d)", text)
    if hm:
        parts = hm.group(1).split(":")
        norm = f"{int(parts[0]):02d}:{parts[1]}:{parts[2] if len(parts)>2 else '00'}"
        return (0, norm, claim["claim_id"])
    return (1, "99:99:99", claim["claim_id"])


def flat_claim_context(claims: list[dict[str, Any]]) -> str:
    return "\n".join(
        f'[{c["claim_id"]}; source={c["document_id"]}; status={c["status"]}] '
        f'{c["text"]} Evidence: "{c["evidence"]}"'
        for c in claims
    )


def padic_claim_context(claims: list[dict[str, Any]], documents: list[dict[str, Any]] | None = None) -> str:
    """Group a preselected claim set; it intentionally performs no second retrieval."""
    # The selection itself already carries p_adic_path when produced by retrieve_claims.
    # For direct callers that pass un-enriched claims, derive paths from the supplied corpus.
    if claims and "p_adic_path" not in claims[0]:
        claims = enrich_claims(claims, documents)
    groups: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    # Reuse the incoming selection. Its membership does not change between B and C.
    for selected in claims:
        c = selected
        groups[tuple(c["p_adic_path"])].append(c)
    chunks = []
    for path in sorted(groups):
        ordered = sorted(groups[path], key=_event_sort_key)
        chunks.append(f"ВЕТВЬ {list(path)} (x={p_adic_code(path)}):")
        chunks.extend(
            f'[{c["claim_id"]}; source={c["document_id"]}; status={c["status"]}] '
            f'{c["text"]} Evidence: "{c["evidence"]}"'
            for c in ordered
        )
    return "\n".join(chunks)


def full_context(documents: list[dict[str, Any]] | None = None,
                 claims: list[dict[str, Any]] | None = None) -> str:
    documents = documents if documents is not None else load_documents()
    claims = claims if claims is not None else load_claims()
    doc_text = "\n\n".join(f'[{d["document_id"]}] {d["title"]} ({d["source_type"]})\n{d["text"]}' for d in documents)
    # Each source document is provided verbatim above. The registry keeps IDs and short
    # claim labels but intentionally omits duplicate evidence quotes to reduce attention
    # memory in full-context mode. Evidence quotes remain available in claims.jsonl.
    index = "\n".join(f'[{c["claim_id"]}; {c["document_id"]}; {c["status"]}] {c["text"]}' for c in claims)
    return doc_text + "\n\nКОМПАКТНЫЙ РЕЕСТР ID УТВЕРЖДЕНИЙ (источники выше)\n" + index


def build_contexts(question: str, top_k: int = 15,
                   documents: list[dict[str, Any]] | None = None,
                   claims: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    documents = documents if documents is not None else load_documents()
    claims = claims if claims is not None else load_claims()
    top = retrieve_claims(question, claims, documents, top_k=top_k)
    flat = flat_claim_context(top)
    branch = padic_claim_context(top, documents)
    ids_b = [c["claim_id"] for c in top]
    # C is a regrouping, not a second retrieval; verify invariance explicitly.
    ids_c = [line.split("]", 1)[0].lstrip("[").split(";", 1)[0] for line in branch.splitlines() if line.startswith("[")]
    if set(ids_b) != set(ids_c) or len(ids_b) != len(ids_c):
        raise AssertionError("B and C must use the identical set of claim IDs")
    return {
        "A": {"context": full_context(documents, claims), "document_ids": [d["document_id"] for d in documents], "claim_ids": [c["claim_id"] for c in claims]},
        "B": {"context": flat, "document_ids": [], "claim_ids": ids_b, "retrieval": top},
        "C": {"context": branch, "document_ids": [], "claim_ids": ids_c, "retrieval": top},
    }
