"""Run the required A/B/C comparison for all 12 questions with one LLM."""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from common import ROOT, load_claims, load_documents, load_questions
from context_builder import build_contexts
from llm import generate_raw_answer, backend_label, DEFAULT_SEED, request_nonce
from health import degeneracy_report
from schemas import parse_model_answer, parse_model_answer_with_safe_repair

RESULTS = ROOT / "results"
RUNS_PATH = RESULTS / "llm_runs.jsonl"
MANIFEST_PATH = RESULTS / "experiment_manifest.json"


def append_jsonl(path: Path, record: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_all(api_key: str | None = None, model: str | None = None, temperature: float = 0.0,
            overwrite: bool = False, limit_questions: int | None = None) -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    # Local Ollama/Transformers inference: no API key or hosted API is required.
    key = None
    from llm import selected_model_name
    questions, claims, docs = load_questions(), load_claims(), load_documents()
    if limit_questions:
        questions = questions[:limit_questions]
    modes = ["A", "B", "C"]
    manifest = {
        "status": "running",
        "created_at_local": datetime.now(ZoneInfo("Europe/Berlin")).isoformat(timespec="seconds"),
        "n_documents": len(docs), "n_claims": len(claims), "n_questions": len(questions),
        "planned_model_calls": len(questions) * len(modes),
        "model": selected_model_name(model),
        "temperature": temperature,
        "seed": DEFAULT_SEED if backend_label() == "ollama_json_schema" else None,
        "generation_mode": "temperature_zero_greedy" if temperature == 0.0 else "sampling",
        "system_prompt_sha256": hashlib.sha256((ROOT / "prompts/system_prompt.md").read_bytes()).hexdigest(),
        "user_prompt_template_sha256": hashlib.sha256((ROOT / "prompts/user_prompt_template.md").read_bytes()).hexdigest(),
        "context_policy": "A=all documents plus full claim reference index; B=BM25 top-15 claims; C=same 15 claims grouped by p-adic path and sorted by time",
        "backend": backend_label(),
        "api_key_required": False,
    }
    if overwrite:
        RUNS_PATH.write_text("", encoding="utf-8")
    elif not RUNS_PATH.exists():
        RUNS_PATH.write_text("", encoding="utf-8")
    valid_ids = {c["claim_id"] for c in claims}
    completed, errors = 0, 0
    statuses: list[str] = []
    for q in questions:
        contexts = build_contexts(q["question"], top_k=15, documents=docs, claims=claims)
        # Fairness assertion: B and C have exactly the same 15 IDs, in membership not order.
        if set(contexts["B"]["claim_ids"]) != set(contexts["C"]["claim_ids"]):
            raise AssertionError(f"B/C claim set mismatch for {q['question_id']}")
        for mode_name in modes:
            ctx = contexts[mode_name]
            record = {
                "run_id": f"{q['question_id']}_{mode_name}",
                "question_id": q["question_id"], "question_type": q["question_type"],
                "mode": mode_name, "question": q["question"],
                "gold_status": q["gold_status"],
                "gold_evidence_ids": q["gold_evidence_ids"],
                "gold_counterevidence_ids": q["gold_counterevidence_ids"],
                "context_claim_ids": ctx["claim_ids"], "context_document_ids": ctx["document_ids"],
                "context_sha256": hashlib.sha256(ctx["context"].encode()).hexdigest(),
                "context_text": ctx["context"],
                "system_prompt_sha256": manifest["system_prompt_sha256"],
                "user_prompt_template_sha256": manifest["user_prompt_template_sha256"],
                "request_nonce": request_nonce(q["question_id"], q["question"], ctx["context"]),
                "temperature": temperature, "seed": manifest["seed"],
                "generation_mode": manifest["generation_mode"], "requested_model": manifest["model"],
            }
            available_ids = set(ctx["claim_ids"]) if mode_name in {"B", "C"} else valid_ids
            try:
                raw = generate_raw_answer(
                    q["question_id"], q["question"], ctx["context"],
                    api_key=key, model=model, temperature=temperature,
                    valid_claim_ids=available_ids,
                )
                record["raw_response"] = raw["raw_text"]
                record.update({k: v for k, v in raw.items() if k != "raw_text"})
                if mode_name == "A" and raw.get("input_truncated"):
                    raise ValueError(
                        "Mode A full-context trial invalid: source context was truncated "
                        f"({raw.get('original_prompt_tokens')} -> {raw.get('prompt_tokens')} tokens). "
                        "Use a GPU/runtime with enough memory; do not score it as full-context A."
                    )
                try:
                    parsed = parse_model_answer(raw["raw_text"], available_ids, expected_question_id=q["question_id"])
                except Exception as first_validation_error:
                    # One constrained correction attempt is cheaper than silently dropping a malformed answer.
                    record["first_validation_error"] = f"{type(first_validation_error).__name__}: {first_validation_error}"
                    record["first_raw_response"] = raw["raw_text"]
                    raw_retry = generate_raw_answer(
                        q["question_id"], q["question"], ctx["context"],
                        api_key=key, model=model, temperature=temperature,
                        valid_claim_ids=available_ids,
                        repair_instruction=(
                            f"Your previous JSON failed validation: {first_validation_error}. "
                            "Correct that issue. Keep answer as a short string; status must be lowercase. "
                            "For both, use at least one available claim ID in each evidence list. "
                            "For insufficient, start the answer with 'Недостаточно данных:' and name the missing information."
                        ),
                    )
                    if mode_name == "A" and raw_retry.get("input_truncated"):
                        raise ValueError("Mode A retry was truncated; cannot score as full-context A")
                    record["first_raw_response"] = record.get("raw_response")
                    record["raw_response"] = raw_retry["raw_text"]
                    record["retry_metadata"] = {k: v for k, v in raw_retry.items() if k != "raw_text"}
                    record["retry_count"] = 1
                    record["total_tokens"] = int(record.get("total_tokens", 0)) + int(raw_retry.get("total_tokens", 0))
                    record["elapsed_seconds"] = round(float(record.get("elapsed_seconds", 0)) + float(raw_retry.get("elapsed_seconds", 0)), 4)
                    try:
                        parsed = parse_model_answer(raw_retry["raw_text"], available_ids, expected_question_id=q["question_id"])
                        record["validation_status"] = "valid_after_repair"
                        record["native_model_output_valid"] = False
                        record["safe_fallback_used"] = False
                    except Exception as second_validation_error:
                        parsed, repair_reasons, safe_fallback = parse_model_answer_with_safe_repair(
                            raw_retry["raw_text"], available_ids, expected_question_id=q["question_id"]
                        )
                        record["post_validation_repair_reasons"] = repair_reasons
                        record["safe_fallback_used"] = safe_fallback
                        record["native_model_output_valid"] = False
                        record["second_validation_error"] = f"{type(second_validation_error).__name__}: {second_validation_error}"
                        record["validation_status"] = "safe_fallback" if safe_fallback else "postprocessed_after_retry"
                record["prediction"] = parsed.model_dump(mode="json")
                statuses.append(parsed.status.value)
                if record.get("validation_status") not in {"safe_fallback", "valid_after_repair", "postprocessed_after_retry"}:
                    record["validation_status"] = "valid"
                record.setdefault("native_model_output_valid", True)
                record.setdefault("safe_fallback_used", False)
            except Exception as exc:
                record["validation_status"] = "error"
                record["error"] = f"{type(exc).__name__}: {exc}"
                errors += 1
            append_jsonl(RUNS_PATH, record)
            completed += 1
            print(f"Base experiment progress: {completed}/{len(questions) * len(modes)} ({q['question_id']}/{mode_name})", flush=True)
    manifest.update({"status": "completed_with_errors" if errors else "completed", "calls_written": completed, "failed_or_invalid": errors})
    manifest["status_distribution"] = degeneracy_report(statuses)
    if manifest["status_distribution"]["degenerate"]:
        manifest["degenerate_warning"] = True
        print("ВНИМАНИЕ: вырожденный прогон, один статус занимает больше 70% ответов:", manifest["status_distribution"], flush=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit-questions", type=int, default=None, help="For smoke testing only; omit for full 12-question run.")
    args = parser.parse_args()
    print(json.dumps(run_all(model=args.model, temperature=args.temperature, overwrite=args.overwrite, limit_questions=args.limit_questions), ensure_ascii=False, indent=2))
