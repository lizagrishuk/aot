"""Executable contract for rubric section 9 without a Streamlit browser.

This exercises the original context transformations against all 120 genuine
recorded LLM inputs. A live Ollama/WebSocket/browser run remains separate.
"""
from pathlib import Path

from common import load_claims, load_documents, load_questions
from stand_logic import apply_intervention, base_packs, load_archives, matched_result
from stress_tests import REMOVALS

ROOT = Path(__file__).resolve().parents[1]


def test_all_120_archived_baseline_and_stress_inputs_match_exactly():
    a = load_archives(ROOT)
    docs, claims, questions = load_documents(), load_claims(), load_questions()
    assert len(a['base_runs']) == 36
    assert len(a['stress_runs']) == 84
    checked = 0
    for q in questions:
        packs = base_packs(q, docs, claims, a)
        assert set(packs['B']['claim_ids']) == set(packs['C']['claim_ids'])
        for mode in 'ABC':
            for kind in ('none', 'permutation', 'remove_premise', 'add_contradiction'):
                if kind == 'remove_premise' and q['question_id'] not in REMOVALS:
                    continue
                result = apply_intervention(q, mode, packs[mode]['context'], docs, claims, kind)
                match, why = matched_result(a, q['question_id'], mode, result, True)
                assert match is not None, (q['question_id'], mode, kind, why)
                checked += 1
    assert checked == 120


def test_custom_claim_removal_is_a_real_transformation():
    a = load_archives(ROOT)
    docs, claims, questions = load_documents(), load_claims(), load_questions()
    q = questions[0]
    for mode in 'ABC':
        original = base_packs(q, docs, claims, a)[mode]
        cid = original['claim_ids'][0]
        modified = apply_intervention(q, mode, original['context'], docs, claims,
                                      'remove_premise', prepared_removal=False,
                                      selected_claim_id=cid)
        assert modified['context_changed']
        assert cid not in modified['claim_ids']
        match, note = matched_result(a, q['question_id'], mode, modified, True)
        assert match is None  # never substitute an old model answer for a new manual removal


def test_uploaded_dossier_does_not_inherit_archived_answers():
    a = load_archives(ROOT)
    docs, claims, questions = load_documents(), load_claims(), load_questions()
    q = questions[0]
    old = base_packs(q, docs, claims, a)['A']['context']
    result = apply_intervention(q, 'A', old, docs, claims)
    assert matched_result(a, q['question_id'], 'A', result, False)[0] is None
