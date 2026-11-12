import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import evaluation as evaluate  # noqa: E402


def test_ece_uses_actual_confidence_intervals_not_rounded_tenths():
    questions = {
        "Q01": {"question_id": "Q01", "question_type": "direct", "gold_status": "entailed",
                "gold_evidence_ids": ["C01"], "gold_counterevidence_ids": [], "gold_answer": "Да"},
        "Q02": {"question_id": "Q02", "question_type": "direct", "gold_status": "entailed",
                "gold_evidence_ids": ["C02"], "gold_counterevidence_ids": [], "gold_answer": "Да"},
    }
    rows = [
        {"question_id": "Q01", "validation_status": "valid", "prediction": {
            "status": "entailed", "answer": "Да", "evidence_ids": ["C01"], "counterevidence_ids": [],
            "confidence": 0.59}},
        {"question_id": "Q02", "validation_status": "valid", "prediction": {
            "status": "contradicted", "answer": "Нет", "evidence_ids": [], "counterevidence_ids": [],
            "confidence": 0.61}},
    ]
    summary = evaluate._metric_summary(rows, questions, expected_runs=2)
    bins = summary["confidence_calibration_bins"]
    assert [(b["range"], b["n"]) for b in bins] == [("[0.4,0.6)", 1), ("[0.6,0.8)", 1)]
    assert summary["confidence_expected_calibration_error"] == 0.51
    assert summary["confidence_brier_score"] == 0.2701


def test_safe_fallback_is_not_counted_as_raw_model_quality(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "RESULTS", tmp_path)
    monkeypatch.setattr(evaluate, "load_questions", lambda: [
        {
            "question_id": "Q01", "question_type": "direct", "gold_status": "entailed",
            "gold_evidence_ids": ["C01"], "gold_counterevidence_ids": [],
        }
    ])
    valid = {
        "run_id": "Q01_A", "question_id": "Q01", "mode": "A", "validation_status": "valid",
        "prediction": {
            "question_id": "Q01", "status": "entailed", "answer": "18:42:16",
            "evidence_ids": ["C01"], "counterevidence_ids": [], "confidence": 0.9,
            "explanation": "C01 прямо указывает время.",
        },
        "total_tokens": 100, "elapsed_seconds": 1.25,
    }
    fallback = {
        "run_id": "Q01_A_retry", "question_id": "Q01", "mode": "A", "validation_status": "safe_fallback",
        "prediction": {
            "question_id": "Q01", "status": "insufficient",
            "answer": "Недостаточно данных: ответ модели не прошёл проверку.",
            "evidence_ids": [], "counterevidence_ids": [], "confidence": 0.1,
            "explanation": "Недостаточно данных: требуется проверка.",
        },
        "total_tokens": 150, "elapsed_seconds": 2.0,
    }
    (tmp_path / "llm_runs.jsonl").write_text(
        json.dumps(valid, ensure_ascii=False) + "\n" + json.dumps(fallback, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    metrics = evaluate.llm_metrics()
    assert metrics["total_saved_runs"] == 2
    assert metrics["model_valid_runs"] == 1
    assert metrics["safe_fallback_runs"] == 1
    assert metrics["per_mode"]["A"]["n_scored_outputs"] == 1
    assert metrics["per_mode"]["A"]["status_accuracy"] == 1.0
    assert metrics["per_mode_end_to_end"]["A"]["n_scored_outputs"] == 2
    assert metrics["per_mode_end_to_end"]["A"]["status_accuracy"] == 0.5


def test_metrics_report_zero_model_coverage_when_only_fallback_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "RESULTS", tmp_path)
    monkeypatch.setattr(evaluate, "load_questions", lambda: [
        {
            "question_id": "Q01", "question_type": "direct", "gold_status": "entailed",
            "gold_evidence_ids": ["C01"], "gold_counterevidence_ids": [],
        }
    ])
    fallback = {
        "run_id": "Q01_B", "question_id": "Q01", "mode": "B", "validation_status": "safe_fallback",
        "prediction": {
            "question_id": "Q01", "status": "insufficient",
            "answer": "Недостаточно данных: безопасный ответ.",
            "evidence_ids": [], "counterevidence_ids": [], "confidence": 0.1,
            "explanation": "Недостаточно данных: ответ не прошёл проверку.",
        },
    }
    (tmp_path / "llm_runs.jsonl").write_text(json.dumps(fallback, ensure_ascii=False) + "\n", encoding="utf-8")

    metrics = evaluate.llm_metrics()
    assert metrics["model_valid_runs"] == 0
    assert metrics["safe_fallback_runs"] == 1
    assert metrics["per_mode"]["B"]["n_scored_outputs"] == 0
    assert metrics["per_mode"]["B"]["status_accuracy"] is None
    assert metrics["per_mode_end_to_end"]["B"]["n_scored_outputs"] == 1


def test_stress_metrics_separate_model_only_from_safe_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "RESULTS", tmp_path)
    baseline = {
        "question_id": "Q01", "mode": "B", "validation_status": "valid",
        "prediction": {"status": "entailed", "evidence_ids": ["C01"], "counterevidence_ids": []},
    }
    stress_fallback = {
        "question_id": "Q01", "mode": "B", "test_type": "permutation",
        "validation_status": "safe_fallback", "actual_status": "insufficient",
        "expected_stress_status": "entailed",
        "prediction": {"status": "insufficient", "evidence_ids": [], "counterevidence_ids": []},
    }
    (tmp_path / "stress_runs.jsonl").write_text(json.dumps(stress_fallback, ensure_ascii=False) + "\n", encoding="utf-8")
    monkeypatch.setattr(evaluate, "load_runs", lambda: [baseline])

    metrics = evaluate.stress_metrics()
    permutation = metrics["by_test"]["permutation"]
    assert metrics["actual_stress_runs"] == 1
    assert metrics["safe_fallback_outputs"] == 1
    assert metrics["model_valid_stress_runs"] == 0
    assert permutation["end_to_end_comparable_pairs"] == 1
    assert permutation["end_to_end_answer_changed_rate"] == 1.0
    assert permutation["model_only_comparable_pairs"] == 0
    assert permutation["model_only_answer_changed_rate"] is None


def test_permutation_metrics_separate_changed_context_from_same_context_controls(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "RESULTS", tmp_path)
    baseline = [
        {"question_id": "Q01", "mode": mode, "validation_status": "valid",
         "prediction": {"status": "entailed", "evidence_ids": ["C01"], "counterevidence_ids": []}}
        for mode in ("A", "B", "C")
    ]
    stress = [
        {"question_id": "Q01", "mode": "A", "test_type": "permutation", "context_changed": True,
         "validation_status": "valid", "actual_status": "contradicted", "expected_stress_status": "entailed",
         "prediction": {"status": "contradicted", "evidence_ids": ["C02"], "counterevidence_ids": []}},
        {"question_id": "Q01", "mode": "B", "test_type": "permutation", "context_changed": False,
         "validation_status": "valid", "actual_status": "insufficient", "expected_stress_status": "entailed",
         "prediction": {"status": "insufficient", "evidence_ids": [], "counterevidence_ids": []}},
        {"question_id": "Q01", "mode": "C", "test_type": "permutation", "context_changed": False,
         "validation_status": "valid", "actual_status": "entailed", "expected_stress_status": "entailed",
         "prediction": {"status": "entailed", "evidence_ids": ["C01"], "counterevidence_ids": []}},
    ]
    (tmp_path / "stress_runs.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in stress) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(evaluate, "load_runs", lambda: baseline)

    metrics = evaluate.stress_metrics()
    permutation = metrics["by_test"]["permutation"]
    assert permutation["document_permutation_context_changed_cases"] == 1
    assert permutation["same_context_replication_control_cases"] == 2
    assert permutation["end_to_end_answer_changed_rate"] == 1.0
    assert permutation["same_context_control_end_to_end_status_changed_rate"] == 0.5


def test_postprocessed_after_retry_is_end_to_end_only(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "RESULTS", tmp_path)
    monkeypatch.setattr(evaluate, "load_questions", lambda: [
        {
            "question_id": "Q01", "question_type": "direct", "gold_status": "entailed",
            "gold_evidence_ids": ["C01"], "gold_counterevidence_ids": [],
        }
    ])
    postprocessed = {
        "run_id": "Q01_C", "question_id": "Q01", "mode": "C", "validation_status": "postprocessed_after_retry",
        "prediction": {
            "question_id": "Q01", "status": "entailed", "answer": "18:42:16",
            "evidence_ids": ["C01"], "counterevidence_ids": [], "confidence": 0.8,
            "explanation": "C01 прямо указывает время.",
        },
        "total_tokens": 180, "elapsed_seconds": 2.5,
    }
    (tmp_path / "llm_runs.jsonl").write_text(json.dumps(postprocessed, ensure_ascii=False) + "\n", encoding="utf-8")
    metrics = evaluate.llm_metrics()
    assert metrics["model_valid_runs"] == 0
    assert metrics["postprocessed_after_retry_runs"] == 1
    assert metrics["per_mode"]["C"]["n_scored_outputs"] == 0
    assert metrics["per_mode_end_to_end"]["C"]["n_scored_outputs"] == 1
