"""Materialize the exact 36 baseline contexts used for A/B/C evaluation."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from common import ROOT, load_questions, load_documents, load_claims
from context_builder import build_contexts

OUT = ROOT / 'results' / 'base_contexts.jsonl'

def main() -> int:
    docs, claims, questions = load_documents(), load_claims(), load_questions()
    records = []
    for q in questions:
        packs = build_contexts(q['question'], top_k=15, documents=docs, claims=claims)
        if set(packs['B']['claim_ids']) != set(packs['C']['claim_ids']):
            raise AssertionError(f'B/C claim ID mismatch for {q["question_id"]}')
        for mode in ('A', 'B', 'C'):
            pack = packs[mode]
            context = pack['context']
            records.append({
                'question_id': q['question_id'], 'question_type': q['question_type'],
                'mode': mode, 'question': q['question'],
                'context_sha256': hashlib.sha256(context.encode('utf-8')).hexdigest(),
                'document_ids': pack['document_ids'], 'claim_ids': pack['claim_ids'],
                'context_text': context,
                'gold_status_excluded': True,
                'note': 'Baseline context only; gold answers and gold statuses are not included in model context.'
            })
    assert len(records) == len(questions) * 3 == 36
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), encoding='utf-8')
    print(json.dumps({'contexts_written': len(records), 'file': str(OUT.relative_to(ROOT)),
                      'bytes': OUT.stat().st_size,
                      'B_C_ids_same_for_all_questions': all(
                          set(next(r for r in records if r['question_id']==q['question_id'] and r['mode']=='B')['claim_ids']) ==
                          set(next(r for r in records if r['question_id']==q['question_id'] and r['mode']=='C')['claim_ids'])
                          for q in questions)}, ensure_ascii=False, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
