import json, re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
rows=[json.loads(x) for x in (ROOT/"data/documents.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
assert 8<=len(rows)<=10,f"Expected 8–10 documents, got {len(rows)}"
ids=[d["document_id"] for d in rows]
assert len(ids)==len(set(ids)),"Duplicate document IDs"
required={"document_id","title","source_type","created_at","author","text"}
assert all(required<=d.keys() for d in rows),"Missing required fields"
words=sum(len(re.findall(r"\b[\w@.-]+\b",d["text"],re.UNICODE)) for d in rows)
assert 5000<=words<=8000,f"Dossier needs 5000–8000 words, currently {words}"
by_id={d["document_id"]:d for d in rows}
assert sum(d["source_type"]=="statement" for d in rows)>=3,"Need at least three witness statements"
assert any(d["source_type"]=="chat" for d in rows),"Missing chat excerpt"
assert any(d["source_type"]=="system_log" for d in rows),"Missing system log"
assert any(d["source_type"]=="policy" for d in rows),"Missing policy/exception document"
assert sum(d["source_type"]=="irrelevant_document" for d in rows)>=2 and {"D09","D10"}<=set(by_id),"Expected two distractor documents (D09 and D10)"
assert "Пользуюсь аккаунтом" in by_id["D03"]["text"] or "я написал одно сообщение" in by_id["D03"]["text"]
assert "24 сообщения" in by_id["D04"]["text"] and "предварительным прочтением" in by_id["D04"]["text"]
print(f"OK: {len(rows)} documents; {words} words; required dossier components are present.")
