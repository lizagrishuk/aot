"""Проверка на вырожденные ответы: модель, которая почти всегда отвечает одним статусом, ничего не измеряет."""
from __future__ import annotations
from collections import Counter

STATUSES = ("entailed", "contradicted", "both", "insufficient")


def degeneracy_report(statuses: list[str], threshold: float = 0.7, min_n: int = 4) -> dict:
    """Вырожденным считается набор из min_n и более ответов, где один статус занимает больше threshold долей."""
    n = len(statuses)
    counts = Counter(statuses)
    if n == 0:
        return {"n": 0, "counts": {}, "majority_status": None, "majority_share": None, "degenerate": False, "threshold": threshold}
    top, top_n = counts.most_common(1)[0]
    share = top_n / n
    return {"n": n, "counts": dict(counts), "majority_status": top, "majority_share": round(share, 3),
            "degenerate": n >= min_n and share > threshold, "threshold": threshold}


def majority_baseline_accuracy(gold_statuses: list[str]) -> float | None:
    """Точность предсказателя-константы, всегда называющего самый частый эталонный статус."""
    if not gold_statuses:
        return None
    return round(Counter(gold_statuses).most_common(1)[0][1] / len(gold_statuses), 4)


def health_gate(predicted: list[str], gold: list[str], margin: float = 0.1) -> dict:
    """Блокирует полный прогон, только если ответы однотипны И точность не лучше константы.

    Одного «>70% одного статуса» мало: при эталоне, где entailed занимает 50% вопросов,
    нормальная модель может дать 73% entailed и всё равно быть лучше константы.
    """
    valid = [(p, g) for p, g in zip(predicted, gold) if p != "INVALID"]
    rep = degeneracy_report([p for p, _ in valid])
    acc = sum(p == g for p, g in zip(predicted, gold)) / len(gold) if gold else 0.0
    baseline = majority_baseline_accuracy(gold) or 0.0
    degenerate = bool(rep["degenerate"]) and acc <= baseline + margin
    enough_valid = len(valid) >= len(gold) // 2
    return {"report": rep, "accuracy": round(acc, 4), "baseline": baseline, "degenerate": degenerate,
            "enough_valid": enough_valid, "ok": (not degenerate) and enough_valid}
