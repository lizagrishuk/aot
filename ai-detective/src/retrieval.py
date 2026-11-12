"""BM25 retrieval for documents or atomic claims, preferring rank-bm25 when installed."""
from __future__ import annotations
import os, re, math
from collections import Counter
from typing import Any

TOKEN_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9_@.-]+")
try:
    from nltk.stem.snowball import SnowballStemmer
    _RU_STEMMER=SnowballStemmer("russian")
    _EN_STEMMER=SnowballStemmer("english")
except ImportError:
    _RU_STEMMER=_EN_STEMMER=None

def tokenize(text: str) -> list[str]:
    if _RU_STEMMER is None and os.getenv("ALLOW_NO_STEMMER") != "1":
        # Без стеммера BM25 даёт другие top-15, и B/C перестают совпадать с прогоном, на котором получены результаты.
        raise RuntimeError("nltk не установлен: pip install nltk (стемминг обязателен для воспроизводимости retrieval)")
    tokens=[]
    for raw in TOKEN_RE.findall(text):
        token=raw.lower()
        if _RU_STEMMER and re.search(r"[а-яё]",token):
            token=_RU_STEMMER.stem(token)
        elif _EN_STEMMER and token.isalpha() and len(token)>4:
            token=_EN_STEMMER.stem(token)
        tokens.append(token)
    return tokens

def _fallback_bm25(corpus: list[list[str]], query_tokens: list[str]) -> list[float]:
    """Deterministic BM25 fallback, used only when rank-bm25 is not installed."""
    n=len(corpus); avgdl=sum(map(len,corpus))/max(n,1)
    df=Counter(token for doc in corpus for token in set(doc))
    scores=[]; k1,b=1.5,0.75
    for doc in corpus:
        counts=Counter(doc); score=0.0; dl=len(doc)
        for token in set(query_tokens):
            freq=counts.get(token,0)
            if not freq:continue
            idf=math.log(1+(n-df[token]+0.5)/(df[token]+0.5))
            denom=freq+k1*(1-b+b*dl/max(avgdl,1))
            score+=idf*(freq*(k1+1)/denom)
        scores.append(score)
    return scores

def bm25_rank(query: str, documents: list[dict[str, Any]], top_k: int = 15) -> list[dict[str, Any]]:
    if not documents or top_k <= 0:return []
    corpus=[tokenize(d.get("text", "")) for d in documents]
    qtokens=tokenize(query)
    try:
        from rank_bm25 import BM25Okapi
        model=BM25Okapi(corpus)
        scores=[float(x) for x in model.get_scores(qtokens)]
    except ImportError:
        scores=_fallback_bm25(corpus,qtokens)
    ranked=sorted(zip(documents,scores),key=lambda x:(-x[1],x[0].get("claim_id",x[0].get("document_id",""))))
    return [{**doc,"_score":float(score)} for doc,score in ranked[:min(top_k,len(ranked))]]

def context_from_documents(docs: list[dict[str, Any]]) -> str:
    return "\n\n".join(f"[{d['document_id']}] {d['title']}\n{d['text']}" for d in docs)

def context_from_claims(claims: list[dict[str, Any]]) -> str:
    return "\n".join(f"[{c['claim_id']}; {c['status']}; {c['document_id']}] {c['text']} Evidence: {c['evidence']}" for c in claims)
