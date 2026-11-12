import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from common import load_documents, load_claims, load_questions
from retrieval import bm25_rank
from context_builder import build_contexts
from padic import p_adic_path, p_adic_code, p_adic_distance

def test_bm25_returns_no_more_than_top_k():
    assert len(bm25_rank("time block message",load_claims(),15))<=15

def test_bm25_handles_empty_corpus():
    assert bm25_rank("query",[],15)==[]

def test_identical_padic_paths_have_zero_distance():
    assert p_adic_distance((1,2,0),(1,2,0),3)==0

def test_padic_distance_differs_at_each_level():
    origin=(0,0,0)
    assert p_adic_distance(origin,(1,0,0),3)==1
    assert p_adic_distance(origin,(0,1,0),3)==1/3
    assert p_adic_distance(origin,(0,0,1),3)==1/9

def test_ultrametric_inequality_for_all_length_three_paths():
    paths=[(a,b,c) for a in range(3) for b in range(3) for c in range(3)]
    for x in paths:
        for y in paths:
            for z in paths:
                assert p_adic_distance(x,z,3)<=max(p_adic_distance(x,y,3),p_adic_distance(y,z,3))+1e-12

def test_padic_code_uses_specified_base_three_formula():
    assert p_adic_code((2,1,2),3)==2+1*3+2*9

def test_padic_paths_depend_on_source_status_and_time():
    docs={d["document_id"]:d for d in load_documents()}
    claims=load_claims(); c=next(x for x in claims if x["claim_id"]=="C26")
    path=p_adic_path(c,docs[c["document_id"]])
    assert len(path)==3 and all(0<=v<=2 for v in path)
    assert path[0]==0  # system/documental source

def test_b_and_c_use_identical_claim_ids():
    for q in load_questions():
        ctx=build_contexts(q["question"],15,load_documents(),load_claims())
        assert len(ctx["B"]["claim_ids"])<=15
        assert len(ctx["C"]["claim_ids"])<=15
        assert set(ctx["B"]["claim_ids"])==set(ctx["C"]["claim_ids"])
