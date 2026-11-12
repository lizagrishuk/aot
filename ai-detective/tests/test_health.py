import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from health import degeneracy_report, majority_baseline_accuracy, health_gate


def test_constant_predictions_are_degenerate():
    r = degeneracy_report(["contradicted"] * 12)
    assert r["degenerate"] and r["majority_status"] == "contradicted" and r["majority_share"] == 1.0


def test_diverse_predictions_are_not_degenerate():
    r = degeneracy_report(["entailed"] * 5 + ["contradicted"] * 3 + ["both", "insufficient"] * 2)
    assert not r["degenerate"]


def test_threshold_boundary_and_small_samples():
    assert not degeneracy_report(["a"] * 7 + ["b"] * 3)["degenerate"]  # ровно 70%, не больше
    assert degeneracy_report(["a"] * 8 + ["b"] * 2)["degenerate"]
    assert not degeneracy_report(["a", "a"])["degenerate"]  # меньше min_n
    assert degeneracy_report([])["n"] == 0


def test_majority_baseline_matches_the_observed_failure():
    gold = ["entailed"] * 6 + ["contradicted"] * 3 + ["insufficient"] * 2 + ["both"]
    assert majority_baseline_accuracy(gold) == 0.5
    # константа «contradicted» набирает ровно долю таких вопросов: 3 из 12 = 0.25
    assert sum(g == "contradicted" for g in gold) / len(gold) == 0.25


def test_gate_passes_skewed_but_better_than_constant():
    gold = ["entailed"] * 6 + ["contradicted"] * 3 + ["both"] + ["insufficient"] * 2
    pred = ["entailed"] * 6 + ["contradicted", "entailed", "contradicted", "INVALID", "insufficient", "entailed"]
    assert health_gate(pred, gold)["ok"]


def test_gate_blocks_constant_prediction():
    gold = ["entailed"] * 6 + ["contradicted"] * 3 + ["both"] + ["insufficient"] * 2
    assert not health_gate(["contradicted"] * 12, gold)["ok"]
