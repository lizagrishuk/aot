import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
rows=[json.loads(x) for x in (ROOT/"data/questions.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
allowed={"entailed","contradicted","both","insufficient"}
types={"direct","multi_hop","temporal","rule_exception","contradiction","insufficient"}
assert len(rows)==12,f"Expected 12 questions, got {len(rows)}"
assert len({q["question_id"] for q in rows})==12,"Duplicate question IDs"
assert all(q["gold_status"] in allowed for q in rows),"Invalid status"
assert all(q["question_type"] in types for q in rows),"Invalid type"
counts={t:sum(q["question_type"]==t for q in rows) for t in types}
assert all(counts[t]==2 for t in types),f"Expected two per type: {counts}"
claim_ids={json.loads(x)["claim_id"] for x in (ROOT/"data/claims.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()}
for q in rows:
    refs=q["gold_evidence_ids"]+q["gold_counterevidence_ids"]
    assert set(refs)<=claim_ids,f'{q["question_id"]} has unknown claim IDs: {set(refs)-claim_ids}'
    if q["gold_status"]=="both":
        assert q["gold_evidence_ids"] and q["gold_counterevidence_ids"],f'{q["question_id"]} status both needs both ID lists'
print("OK: 12 questions; 2 each of 6 types; valid status labels and existing evidence IDs.")
