import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_final_submission_archive_is_complete():
    required = [
        RESULTS / "llm_runs.jsonl",
        RESULTS / "stress_runs.jsonl",
        RESULTS / "stress_cases.jsonl",
        RESULTS / "llm_metrics.json",
        RESULTS / "stress_metrics.json",
        RESULTS / "retrieval_metrics.json",
        ROOT / "report.md",
        ROOT / "report.pdf",
    ]
    if not all(p.exists() for p in required):
        pytest.skip("Clean source archive has no completed experiment results yet")

    base = _jsonl(RESULTS / "llm_runs.jsonl")
    stress = _jsonl(RESULTS / "stress_runs.jsonl")
    cases = _jsonl(RESULTS / "stress_cases.jsonl")
    assert len(base) == 36
    assert len(stress) == 84
    assert len(cases) == 84
    assert len({(r["question_id"], r["mode"]) for r in base}) == 36
    assert len({(r["question_id"], r["mode"], r["test_type"]) for r in stress}) == 84

    lm = json.loads((RESULTS / "llm_metrics.json").read_text(encoding="utf-8"))
    sm = json.loads((RESULTS / "stress_metrics.json").read_text(encoding="utf-8"))
    rm = json.loads((RESULTS / "retrieval_metrics.json").read_text(encoding="utf-8"))
    assert lm["total_saved_runs"] == 36
    assert sm["actual_stress_runs"] == 84
    assert rm["B_C_same_claim_ids_all_questions"] is True
    assert {r["model"] for r in base + stress} == {"qwen2.5:7b-instruct"}

    report = (ROOT / "report.md").read_text(encoding="utf-8")
    assert len(re.findall(r"\w+", report)) <= 1500
    assert (ROOT / "report.pdf").stat().st_size > 0
