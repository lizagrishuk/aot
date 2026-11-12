"""Reproducible retrieval, model-only, end-to-end, and stress metrics."""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
from common import ROOT, load_documents, load_questions, load_claims
from context_builder import build_contexts

RESULTS = ROOT / "results"
# Only answers that pass the strict parser directly count as LLM-quality outputs.
# A deterministic repair can yield a safe usable application output, but it is tracked separately.
MODEL_STATUSES = {"valid", "valid_after_repair"}
POSTPROCESSED_STATUSES = {"postprocessed_after_retry"}
USABLE_STATUSES = MODEL_STATUSES | POSTPROCESSED_STATUSES | {"safe_fallback"}


def _prf(predicted: set[str], gold: set[str]) -> dict[str, float]:
    tp = len(predicted & gold)
    precision = tp / len(predicted) if predicted else (1.0 if not gold else 0.0)
    recall = tp / len(gold) if gold else (1.0 if not predicted else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def polarity(text: str | None) -> str | None:
    """'yes'/'no' if the answer starts with «Да»/«Нет», otherwise None."""
    first = (text or "").strip().lower().lstrip("«\"'(").split(" ")[0].strip(".,:;!—-")
    return {"да": "yes", "нет": "no"}.get(first)


def retrieval_metrics() -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    docs, claims, questions = load_documents(), load_claims(), load_questions()
    rows = []
    for q in questions:
        contexts = build_contexts(q["question"], top_k=15, documents=docs, claims=claims)
        expected = set(q["gold_evidence_ids"] + q["gold_counterevidence_ids"])
        b, c = set(contexts["B"]["claim_ids"]), set(contexts["C"]["claim_ids"])
        all_ids = {claim["claim_id"] for claim in claims}
        relevant_docs = {x["document_id"] for x in claims if x["claim_id"] in expected}
        b_docs = {x["document_id"] for x in claims if x["claim_id"] in b}
        c_docs = {x["document_id"] for x in claims if x["claim_id"] in c}
        rows.append({
            "question_id": q["question_id"], "question_type": q["question_type"], "gold_status": q["gold_status"],
            "A_full_context_claim_recall": 1.0 if expected <= all_ids else 0.0,
            "B_bm25_top15_evidence_recall": round(len(expected & b) / len(expected), 4) if expected else 1.0,
            "C_padic_top15_evidence_recall": round(len(expected & c) / len(expected), 4) if expected else 1.0,
            "B_document_source_recall": round(len(relevant_docs & b_docs) / len(relevant_docs), 4) if relevant_docs else 1.0,
            "C_document_source_recall": round(len(relevant_docs & c_docs) / len(relevant_docs), 4) if relevant_docs else 1.0,
            "B_claim_ids": contexts["B"]["claim_ids"], "C_claim_ids": contexts["C"]["claim_ids"],
            "B_C_same_claim_ids": set(contexts["B"]["claim_ids"]) == set(contexts["C"]["claim_ids"]),
        })
    n = max(len(rows), 1)
    result = {
        "n_documents": len(docs), "n_claims": len(claims), "n_questions": len(questions), "top_k": 15,
        "A_mean_full_context_evidence_coverage": round(sum(x["A_full_context_claim_recall"] for x in rows) / n, 4),
        "B_mean_gold_evidence_recall": round(sum(x["B_bm25_top15_evidence_recall"] for x in rows) / n, 4),
        "C_mean_gold_evidence_recall": round(sum(x["C_padic_top15_evidence_recall"] for x in rows) / n, 4),
        "B_C_same_claim_ids_all_questions": all(x["B_C_same_claim_ids"] for x in rows),
        "results": rows,
        "note": "Deterministic retrieval coverage only, not LLM accuracy. B and C must use the same claim set; only order/grouping differs.",
    }
    (RESULTS / "retrieval_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def load_runs() -> list[dict]:
    path = RESULTS / "llm_runs.jsonl"
    if not path.exists():
        return []
    output = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                output.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return output


def _has_prediction(row: dict) -> bool:
    return row.get("validation_status") in USABLE_STATUSES and isinstance(row.get("prediction"), dict)


def _metric_summary(rows: list[dict], qmap: dict[str, dict], expected_runs: int = 12) -> dict:
    detailed = []
    for r in rows:
        if not isinstance(r.get("prediction"), dict) or r.get("question_id") not in qmap:
            continue
        pred, q = r["prediction"], qmap[r["question_id"]]
        ev = _prf(set(pred.get("evidence_ids", [])), set(q["gold_evidence_ids"]))
        ce = _prf(set(pred.get("counterevidence_ids", [])), set(q["gold_counterevidence_ids"]))
        status_correct = pred.get("status") == q["gold_status"]
        gold_pol = polarity(q.get("gold_answer"))
        detailed.append({
            "gold_polarity": gold_pol, "predicted_polarity": polarity(pred.get("answer")),
            "question_id": q["question_id"], "question_type": q["question_type"],
            "gold_status": q["gold_status"], "predicted_status": pred.get("status"),
            "status_correct": status_correct,
            "evidence_precision": ev["precision"], "evidence_recall": ev["recall"], "evidence_f1": ev["f1"],
            "counterevidence_precision": ce["precision"], "counterevidence_recall": ce["recall"],
            "counterevidence_f1": ce["f1"], "confidence": pred.get("confidence"),
            "total_tokens": r.get("total_tokens"), "elapsed_seconds": r.get("elapsed_seconds"),
            "validation_status": r.get("validation_status"),
        })
    def avg(key):
        values = [d[key] for d in detailed if isinstance(d.get(key), (int, float))]
        return round(sum(values) / len(values), 4) if values else None
    gold_both = [d for d in detailed if d["gold_status"] == "both"]
    pred_both = [d for d in detailed if d["predicted_status"] == "both"]
    gold_ins = [d for d in detailed if d["gold_status"] == "insufficient"]
    confident_wrong_ins = [d for d in gold_ins if d["predicted_status"] != "insufficient" and (d["confidence"] or 0) >= 0.7]
    confidence_pairs = [(float(d["confidence"]), 1.0 if d["status_correct"] else 0.0)
                        for d in detailed if isinstance(d.get("confidence"), (int, float))]
    brier = ece = None
    calibration_bins = []
    if confidence_pairs:
        brier = round(sum((conf - correct) ** 2 for conf, correct in confidence_pairs) / len(confidence_pairs), 4)
        ece_value = 0.0
        # ECE bins use the actual confidence value, not round(confidence * 10):
        # rounding would incorrectly move e.g. 0.59 into the [0.6, 0.8) bin.
        # Integer tenths define exact interval edges without cumulative float addition.
        for lo10 in (0, 2, 4, 6, 8):
            lo, hi = lo10 / 10, (lo10 + 2) / 10
            bucket = [(c, a) for c, a in confidence_pairs
                      if (lo <= c < hi) or (lo10 == 8 and c == 1.0)]
            if bucket:
                mean_conf = sum(c for c, _ in bucket) / len(bucket)
                acc = sum(a for _, a in bucket) / len(bucket)
                ece_value += len(bucket) / len(confidence_pairs) * abs(mean_conf - acc)
                calibration_bins.append({"range": f"[{lo:.1f},{hi:.1f}{']' if hi >= 1 else ')'}", "n": len(bucket),
                                         "mean_confidence": round(mean_conf, 4), "accuracy": round(acc, 4)})
        ece = round(ece_value, 4)
    yes_no = [d for d in detailed if d["gold_polarity"]]
    return {
        "n_scored_outputs": len(detailed), "expected_runs": expected_runs,
        # Status accuracy follows the team convention (a correct «нет» to a yes/no question has gold status `contradicted`).
        # A model can give the right «нет» and still choose another status label; this metric shows that separately.
        "answer_polarity_n": len(yes_no),
        "answer_polarity_accuracy": round(sum(d["predicted_polarity"] == d["gold_polarity"] for d in yes_no) / len(yes_no), 4) if yes_no else None,
        "coverage_rate": round(len(detailed) / expected_runs, 4) if expected_runs else None,
        "status_accuracy": round(sum(d["status_correct"] for d in detailed) / len(detailed), 4) if detailed else None,
        "confidence_brier_score": brier, "confidence_expected_calibration_error": ece,
        "confidence_calibration_bins": calibration_bins,
        "evidence_precision_macro": avg("evidence_precision"), "evidence_recall_macro": avg("evidence_recall"),
        "evidence_f1_macro": avg("evidence_f1"), "counterevidence_precision_macro": avg("counterevidence_precision"),
        "counterevidence_recall_macro": avg("counterevidence_recall"), "counterevidence_f1_macro": avg("counterevidence_f1"),
        "both_precision": round(sum(d["predicted_status"] == "both" and d["gold_status"] == "both" for d in detailed) / len(pred_both), 4) if pred_both else None,
        "both_recall": round(sum(d["predicted_status"] == "both" and d["gold_status"] == "both" for d in detailed) / len(gold_both), 4) if gold_both else None,
        "insufficient_class_accuracy": round(sum(d["predicted_status"] == "insufficient" for d in gold_ins) / len(gold_ins), 4) if gold_ins else None,
        "confident_wrong_answers_on_insufficient": len(confident_wrong_ins),
        "confident_wrong_rate_on_insufficient": round(len(confident_wrong_ins) / len(gold_ins), 4) if gold_ins else None,
        "mean_total_tokens": avg("total_tokens"), "mean_elapsed_seconds": avg("elapsed_seconds"), "per_question": detailed,
    }


def llm_metrics() -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    questions = load_questions()
    qmap = {q["question_id"]: q for q in questions}
    runs = load_runs()
    completed = [r for r in runs if _has_prediction(r)]
    model_outputs = [r for r in completed if r.get("validation_status") in MODEL_STATUSES]
    native = [r for r in completed if r.get("validation_status") == "valid"]
    repaired = [r for r in completed if r.get("validation_status") == "valid_after_repair"]
    fallbacks = [r for r in completed if r.get("validation_status") == "safe_fallback"]
    postprocessed = [r for r in completed if r.get("validation_status") in POSTPROCESSED_STATUSES]
    failed = [r for r in runs if r not in completed]

    per_mode_model, per_mode_end_to_end = {}, {}
    all_type_model = defaultdict(list)
    all_type_end = defaultdict(list)
    for mode in ("A", "B", "C"):
        mode_all = [r for r in completed if r.get("mode") == mode]
        mode_model = [r for r in mode_all if r.get("validation_status") in MODEL_STATUSES]
        mode_native = [r for r in mode_all if r.get("validation_status") == "valid"]
        mode_repaired = [r for r in mode_all if r.get("validation_status") == "valid_after_repair"]
        mode_fallback = [r for r in mode_all if r.get("validation_status") == "safe_fallback"]
        mode_postprocessed = [r for r in mode_all if r.get("validation_status") in POSTPROCESSED_STATUSES]
        model_summary = _metric_summary(mode_model, qmap, 12)
        end_summary = _metric_summary(mode_all, qmap, 12)
        model_summary.update({"native_model_outputs": len(mode_native), "retry_valid_outputs": len(mode_repaired),
                              "postprocessed_after_retry_outputs": len(mode_postprocessed),
                              "safe_fallback_outputs": len(mode_fallback), "completed_outputs_including_fallback": len(mode_all)})
        end_summary.update({"native_model_outputs": len(mode_native), "retry_valid_outputs": len(mode_repaired),
                            "postprocessed_after_retry_outputs": len(mode_postprocessed),
                            "safe_fallback_outputs": len(mode_fallback), "completed_outputs_including_fallback": len(mode_all)})
        per_mode_model[mode] = model_summary
        per_mode_end_to_end[mode] = end_summary
        for r in mode_model:
            d = _metric_summary([r], qmap, 1)["per_question"]
            if d: all_type_model[d[0]["question_type"]].extend(d)
        for r in mode_all:
            d = _metric_summary([r], qmap, 1)["per_question"]
            if d: all_type_end[d[0]["question_type"]].extend(d)

    def acc_by_type(mapping):
        return {typ: round(sum(x["status_correct"] for x in vals) / len(vals), 4) if vals else None
                for typ, vals in mapping.items()}
    native_acc = {}
    for mode in ("A", "B", "C"):
        subset = [r for r in native if r.get("mode") == mode]
        native_acc[mode] = round(sum(r.get("prediction", {}).get("status") == qmap[r["question_id"]]["gold_status"] for r in subset) / len(subset), 4) if subset else None
    out = {
        "total_saved_runs": len(runs), "valid_runs": len(completed),
        "model_valid_runs": len(model_outputs), "native_model_valid_runs": len(native),
        "retry_valid_runs": len(repaired), "postprocessed_after_retry_runs": len(postprocessed),
        "safe_fallback_runs": len(fallbacks), "invalid_or_failed_runs": len(failed),
        "validation_status_counts": {s: sum(r.get("validation_status") == s for r in runs) for s in ("valid", "valid_after_repair", "postprocessed_after_retry", "safe_fallback", "error")},
        "model_answer_metrics_available": bool(model_outputs), "per_mode": per_mode_model,
        "per_mode_end_to_end": per_mode_end_to_end,
        "status_accuracy_by_question_type": acc_by_type(all_type_model),
        "end_to_end_status_accuracy_by_question_type": acc_by_type(all_type_end),
        "native_model_status_accuracy_by_mode": native_acc,
        "note": "per_mode scores only model generations that passed strict validation on the first or second attempt (`valid`, `valid_after_repair`). Deterministically normalized outputs (`postprocessed_after_retry`) and safe fallbacks are excluded from raw-model metrics and shown separately. per_mode_end_to_end includes all usable outputs and describes the safety-wrapped application, not raw LLM accuracy.",
    }
    from health import degeneracy_report, majority_baseline_accuracy
    pred_status = lambda rows: [r["prediction"]["status"] for r in rows if isinstance(r.get("prediction"), dict)]
    out["status_distribution_by_mode"] = {m: degeneracy_report(pred_status([r for r in completed if r.get("mode") == m])) for m in ("A", "B", "C")}
    out["overall_status_distribution"] = degeneracy_report(pred_status(completed))
    out["gold_majority_baseline_accuracy"] = majority_baseline_accuracy([q["gold_status"] for q in questions])
    out["degenerate_warning"] = bool(out["overall_status_distribution"]["degenerate"])
    if out["degenerate_warning"]:
        out["degenerate_note"] = ("Один статус занимает больше 70% ответов: модель почти не различает статусы. "
                                  "Точность и устойчивость по такому прогону интерпретировать нельзя.")
    (RESULTS / "llm_metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out


def stress_metrics() -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "stress_runs.jsonl"
    baseline_by = {(r.get("question_id"), r.get("mode")): r for r in load_runs() if _has_prediction(r)}
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if not rows:
        out = {"actual_stress_runs": 0, "usable_stress_outputs": 0, "model_valid_stress_runs": 0,
               "safe_fallback_outputs": 0, "metrics_available": False,
               "note": "No actual LLM stress outputs saved. Prepared test cases are not observed results."}
        (RESULTS / "stress_metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return out
    usable = [r for r in rows if r.get("validation_status") in USABLE_STATUSES and isinstance(r.get("prediction"), dict)]
    model_valid = [r for r in usable if r.get("validation_status") in MODEL_STATUSES]
    fallbacks = [r for r in usable if r.get("validation_status") == "safe_fallback"]
    postprocessed = [r for r in usable if r.get("validation_status") in POSTPROCESSED_STATUSES]
    stats = {}
    for typ in ("permutation", "remove_premise", "add_contradiction"):
        all_subset = [r for r in usable if r.get("test_type") == typ]
        model_subset = [r for r in model_valid if r.get("test_type") == typ]
        fallback_count = sum(r.get("validation_status") == "safe_fallback" for r in all_subset)
        if typ == "permutation":
            # Only mode A receives a changed model context: it consumes the source
            # documents in their displayed order. B/C rebuild their retrieval context
            # from the same claim corpus, so a permutation of the source-document list
            # must leave their prompts byte-identical. Those runs are repeat controls
            # and must not be mixed into the document-order sensitivity estimate.
            changed_context_rows = [r for r in all_subset if r.get("context_changed", True)]
            invariant_context_rows = [r for r in all_subset if r.get("context_changed") is False]

            def paired(rows):
                all_pairs, model_pairs = [], []
                for r in rows:
                    base = baseline_by.get((r.get("question_id"), r.get("mode")))
                    if not base:
                        continue
                    pair = (base["prediction"], r["prediction"])
                    all_pairs.append(pair)
                    if base.get("validation_status") in MODEL_STATUSES and r.get("validation_status") in MODEL_STATUSES:
                        model_pairs.append(pair)
                return all_pairs, model_pairs

            changed_all_pairs, changed_model_pairs = paired(changed_context_rows)
            control_all_pairs, control_model_pairs = paired(invariant_context_rows)
            changed_rate = lambda pairs: (round(sum(a.get("status") != b.get("status") for a, b in pairs) / len(pairs), 4) if pairs else None)
            evidence_stability = lambda pairs: (round(sum(set(a.get("evidence_ids", [])) == set(b.get("evidence_ids", [])) for a, b in pairs) / len(pairs), 4) if pairs else None)
            stats[typ] = {
                "observed_outputs": len(all_subset), "native_or_repaired_outputs": len(model_subset), "safe_fallback_outputs": fallback_count,
                "document_permutation_context_changed_cases": len(changed_context_rows),
                "same_context_replication_control_cases": len(invariant_context_rows),
                "end_to_end_comparable_pairs": len(changed_all_pairs),
                "end_to_end_answer_changed_rate": changed_rate(changed_all_pairs),
                "model_only_comparable_pairs": len(changed_model_pairs),
                "model_only_answer_changed_rate": changed_rate(changed_model_pairs),
                "model_only_evidence_ids_stability": evidence_stability(changed_model_pairs),
                "same_context_control_end_to_end_pairs": len(control_all_pairs),
                "same_context_control_end_to_end_status_changed_rate": changed_rate(control_all_pairs),
                "same_context_control_model_only_pairs": len(control_model_pairs),
                "same_context_control_model_only_status_changed_rate": changed_rate(control_model_pairs),
                "same_context_control_model_only_evidence_ids_stability": evidence_stability(control_model_pairs),
            }
        elif typ == "remove_premise":
            expected = [r for r in all_subset if r.get("expected_stress_status") in {"insufficient", "both"}]
            expected_model = [r for r in model_subset if r.get("expected_stress_status") in {"insufficient", "both"}]
            stats[typ] = {
                "observed_outputs": len(all_subset), "native_or_repaired_outputs": len(model_subset), "safe_fallback_outputs": fallback_count,
                "redundant_control_cases": sum(r.get("expected_stress_status") not in {"insufficient", "both"} for r in all_subset),
                "end_to_end_changed_to_insufficient_or_both_rate": round(sum(r.get("actual_status") in {"insufficient", "both"} for r in expected) / len(expected), 4) if expected else None,
                "model_only_changed_to_insufficient_or_both_rate": round(sum(r.get("actual_status") in {"insufficient", "both"} for r in expected_model) / len(expected_model), 4) if expected_model else None,
                "end_to_end_matches_expected_status_rate": round(sum(r.get("actual_status") == r.get("expected_stress_status") for r in all_subset) / len(all_subset), 4) if all_subset else None,
                "model_only_matches_expected_status_rate": round(sum(r.get("actual_status") == r.get("expected_stress_status") for r in model_subset) / len(model_subset), 4) if model_subset else None,
            }
        else:
            exp_both = [r for r in all_subset if r.get("expected_stress_status") == "both"]
            exp_same = [r for r in all_subset if r.get("expected_stress_status") != "both"]
            stats[typ] = {
                "expected_both_cases": len(exp_both),
                "both_recall_on_expected_both": round(sum(r.get("actual_status") == "both" for r in exp_both) / len(exp_both), 4) if exp_both else None,
                "expected_unchanged_cases": len(exp_same),
                "unchanged_rate_when_source_is_weaker": round(sum(r.get("actual_status") == r.get("expected_stress_status") for r in exp_same) / len(exp_same), 4) if exp_same else None,
                "observed_outputs": len(all_subset), "native_or_repaired_outputs": len(model_subset), "safe_fallback_outputs": fallback_count,
                "end_to_end_both_status_rate": round(sum(r.get("actual_status") == "both" for r in all_subset) / len(all_subset), 4) if all_subset else None,
                "model_only_both_status_rate": round(sum(r.get("actual_status") == "both" for r in model_subset) / len(model_subset), 4) if model_subset else None,
                "end_to_end_matches_expected_status_rate": round(sum(r.get("actual_status") == r.get("expected_stress_status") for r in all_subset) / len(all_subset), 4) if all_subset else None,
                "model_only_matches_expected_status_rate": round(sum(r.get("actual_status") == r.get("expected_stress_status") for r in model_subset) / len(model_subset), 4) if model_subset else None,
            }
    out = {
        "actual_stress_runs": len(rows), "usable_stress_outputs": len(usable),
        "model_valid_stress_runs": len(model_valid), "native_model_valid_stress_runs": sum(r.get("validation_status") == "valid" for r in model_valid),
        "retry_valid_stress_runs": sum(r.get("validation_status") == "valid_after_repair" for r in model_valid),
        "postprocessed_after_retry_outputs": len(postprocessed), "safe_fallback_outputs": len(fallbacks),
        "failed_or_invalid": len(rows) - len(usable),
        "metrics_available": bool(model_valid), "by_test": stats,
        "note": "Model-only metrics include only strict parser-valid outputs from the first or second model generation. `postprocessed_after_retry` and `safe_fallback` are excluded from raw-LLM metrics but included in end-to-end metrics. End-to-end metrics describe the complete system, not raw model robustness. Interpret test metrics together with model_valid_stress_runs.",
    }
    (RESULTS / "stress_metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out


def main():
    ret = retrieval_metrics()
    llm = llm_metrics()
    stress = stress_metrics()
    print(json.dumps({"retrieval": {k: v for k, v in ret.items() if k != "results"},
                      "llm": {k: v for k, v in llm.items() if k not in {"per_mode", "per_mode_end_to_end"}},
                      "stress": stress}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
