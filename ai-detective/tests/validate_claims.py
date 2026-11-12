import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
docs={d["document_id"]:d for d in (json.loads(x) for x in (ROOT/"data/documents.jsonl").read_text(encoding="utf-8").splitlines() if x.strip())}
claims=[json.loads(x) for x in (ROOT/"data/claims.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
allowed={"direct","reported","inferred","disputed"}
assert 30<=len(claims)<=40,f"Expected 30–40 claims, got {len(claims)}"
assert len({c["claim_id"] for c in claims})==len(claims),"Duplicate claim IDs"
for c in claims:
    assert {"claim_id","text","document_id","evidence","status"}<=c.keys(),c.get("claim_id")
    assert c["document_id"] in docs,f'Unknown document: {c["document_id"]}'
    assert c["status"] in allowed,f'Invalid status: {c["status"]}'
    assert c["evidence"] in docs[c["document_id"]]["text"],f'Evidence is not exact substring: {c["claim_id"]}'
print(f"OK: {len(claims)} claims; IDs unique; evidence strings are exact substrings in existing documents.")
