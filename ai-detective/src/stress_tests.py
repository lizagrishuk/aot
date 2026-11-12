"""Prepare/run the three required stress tests for all questions and modes.

The permutation intervention changes only the source-document order (seed 42).
Mode A exposes that order directly. Modes B/C are retrieval-derived: their selected
claim set and rendered context are invariant to source-document list order. Those
B/C cases are retained as same-context replication controls, not mislabelled as
claim-line or p-adic-branch permutations.

The selected local LLM backend (Ollama by default, Transformers optionally) runs
without an API key. The first model setup may download public weights.
"""
from __future__ import annotations
import argparse, json, random, hashlib
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any
from common import ROOT, load_documents, load_claims, load_questions
from context_builder import build_contexts, full_context
from llm import generate_raw_answer, DEFAULT_SEED, request_nonce, SYSTEM_PROMPT, USER_TEMPLATE
from schemas import parse_model_answer, parse_model_answer_with_safe_repair
from interventions import remove_fact

RESULTS = ROOT / "results"
# Test 2 (removal) applies only to multi-hop and temporal questions (methodology, stage 8, test 2).
# What is removed is one FACT: every claim and every source sentence that states it (see interventions.py).
# `literals` identify the fact; after the intervention none of them may remain in the context.
# Q03-Q06: remove the necessary underlying fact (including every repeated source statement);
# the status becomes insufficient, not merely a change of one log line.
REMOVALS = {
 "Q03": {"literals": ["самостоятельно менять", "самостоятельно снимать", "изменение статуса ограничения", "не имею доступа к соответствующей кнопке", "требуется проверка модерации", "принимает ответственный специалист модерации"],
         "expected": "insufficient", "reason": "fact removed: the support role cannot change the restriction; without it the question 'why could she not unblock' has no basis"},
 "Q04": {"literals": ["unique_message_count", "одним исходным message_id"],
         "expected": "insufficient", "reason": "fact removed: the 24 events belong to one unique message; the log link between the events and the single message is gone"},
 "Q05": {"literals": ["18:40", "delay_detected", "началась до блокировки"],
         "expected": "insufficient", "reason": "fact removed: the queue delay time and every statement that it began before the restriction"},
 "Q06": {"literals": [
             "19:12",                                      # D02 timeline + D07 account_unrestricted + C29
             "ошибочно указала время снятия блокировки",    # C17 registry
             "по ошибке указала",                           # D05 source and C17 evidence
             "время снятия блокировки 19:10",              # D02 support-summary retelling
             "Позднее выяснилось, что 19:10",              # D05 explicit correction
             "Это время относится к одобрению снятия ограничения, а не к изменению статуса аккаунта", # D02 explicit correction
             "Фраза «блокировку сняли в 19:10»",            # D05 discussion of misdated note
             "здесь это были разные события",              # D05 indirect confirmation actual unblock later
             "Время у них различается",                    # D07 indirect statement of differing operation timestamps
             "Об ошибке во времени",                       # D05 heading identifying false note
             "Ошибку я заметила",                           # D05 note correction
             "Моя ошибка была добросовестной",             # D05 second confirmation that note was wrong
         ],
         "expected": "insufficient", "reason": "remove the single underlying fact: the actual unblock happened AFTER 19:10. Remove its log event, timeline, registry claim and all corrections of the erroneous support timestamp; keep 19:10 approval, which by itself does not establish the actual unblock time"},
}
# Test 3: one incompatible claim from another source, written as an ordinary source record (no "test" banner).
# expected = "both" only when the added source is as reliable as the one it contradicts (no basis to choose);
# when the added source is weaker (witness vs. system log / regulation) the status must not change.
CONTRADICTIONS = {
 "Q01": {"source_type": "system_log", "text": "2026-03-14T18:44:09 — audit_reconciliation incident=INC-260314-17 operation=account_restricted user=usr_4821 transition_ref=the_same_single_restriction_transition_as_D07_at_18:42:16 canonical_event_timestamp=2026-03-14T18:44:09 note=there_was_no_second_restriction_operation", "expected": "both", "reason": "an equally reliable system audit reconciliation assigns a different timestamp to the same single restriction transition; the explicit same-transition reference rules out a second restriction event"},
 "Q02": {"source_type": "system_log", "text": "2026-03-14T19:20:00 — audit_reconciliation incident=INC-260314-17 operation=account_unrestricted user=usr_4821 actor=denis_krylov transition_ref=the_same_single_unrestriction_transition_as_D07_at_19:12:19 canonical_event_timestamp=2026-03-14T19:10:31 note=there_was_no_second_unrestriction_operation", "expected": "both", "reason": "an equally reliable system audit reconciliation assigns a different timestamp to the same single unrestriction transition; the explicit same-transition reference rules out a second unrestriction event"},
 "Q03": {"source_type": "statement", "text": "Участник чата считает, что Марина Белова могла сама снять блокировку.", "expected": "entailed", "reason": "an opinion is weaker than the regulation and Marina's own report, status must not change"},
 "Q04": {"source_type": "statement", "text": "Денис утверждает, что 24 события были 24 отдельными отправками и не связаны с задержкой очереди.", "expected": "entailed", "reason": "witness claim is weaker than the system log (C24-C26), status must not change"},
 "Q05": {"source_type": "system_log", "text": "2026-03-14T18:44:00 — notification_queue incident_audit queue=notify-main incident=INC-260314-17 delay_before_restriction=false first_delay_detected_at=2026-03-14T18:43:20 (до ограничения аккаунта в 18:42:16 задержки очереди не было)", "expected": "both", "reason": "independent system-log record explicitly denies any queue delay before the restriction, while C24 records one at 18:40:03; equally ranked sources make the answer disputed"},
 "Q06": {"source_type": "statement", "text": "Артём утверждает, что аккаунт был фактически разблокирован в 19:10.", "expected": "contradicted", "reason": "witness claim is weaker than the log C29, status must not change"},
 "Q07": {"source_type": "statement", "text": "Участник чата сообщил, что для INC-260314-17 не было ни технического заключения, ни одобрения.", "expected": "entailed", "reason": "unverified chat hearsay is weaker than Ilya’s first-person report of his own technical conclusion, Marina’s confirmation that she linked it in the incident card, and the direct approval log"},
 "Q08": {"source_type": "statement", "text": "Участник чата считает, что инженер мог единолично одобрить снятие ограничения.", "expected": "contradicted", "reason": "opinion is weaker than the regulation C33, status must not change"},
 "Q09": {"source_type": "statement", "text": "Участник чата утверждает, что Артём получил предупреждение о приближении к порогу.", "expected": "both", "reason": "one more unverified statement on the same side; the conflict between C36 and C37 stays unresolved, status must stay both"},
 "Q10": {"source_type": "statement", "text": "Участник чата утверждает, что журнал подтверждает 24 уникальных сообщения.", "expected": "contradicted", "reason": "a retelling is weaker than the log itself (C26), status must not change"},
 "Q11": {"source_type": "statement", "text": "Участник чата утверждает, что Артём намеренно обошёл правила массовой рассылки.", "expected": "insufficient", "reason": "an unverified accusation adds no evidence of intent, status must stay insufficient"},
 "Q12": {"source_type": "statement", "text": "Участник чата утверждает, что Артём использовал скрипт для отправки сообщения.", "expected": "insufficient", "reason": "hearsay against the user's own denial: neither side is verified, status must stay insufficient"},
}

SYNTH_TITLE = {"system_log": "Дополнительная запись журнала", "statement": "Дополнительное свидетельство"}
SYNTH_STATUS = {"system_log": "direct", "statement": "reported"}
SYNTH_PATH = {"system_log": (0, 0, 2), "statement": (1, 1, 2)}


def _add_synthetic(mode: str, original: str, spec: dict) -> tuple[str, dict]:
    """Add CX01 as an ordinary source record. Test marking lives only in case metadata."""
    from padic import p_adic_code
    st = spec["source_type"]; text = spec["text"]; status = SYNTH_STATUS[st]
    claim = {"claim_id": "CX01", "text": text, "document_id": "X01", "status": status, "evidence": text}
    line = f'[CX01; source=X01; status={status}] {text} Evidence: "{text}"'
    if mode == "A":
        marker = "\n\nКОМПАКТНЫЙ РЕЕСТР ID УТВЕРЖДЕНИЙ"
        head, tail = original.split(marker, 1)
        doc = f"[X01] {SYNTH_TITLE[st]} ({st})\n{text}"
        reg = f"[CX01; X01; {status}] {text}"
        return head + "\n\n" + doc + marker + tail + "\n" + reg, claim
    if mode == "B":
        return original + "\n" + line, claim
    path = SYNTH_PATH[st]
    header = f"ВЕТВЬ {list(path)} (x={p_adic_code(path)}):"
    lines = original.splitlines()
    if header in lines:
        i = lines.index(header) + 1
        while i < len(lines) and not lines[i].startswith("ВЕТВЬ "):
            i += 1
        lines.insert(i, line)
        return "\n".join(lines), claim
    return original + "\n" + header + "\n" + line, claim


def jsonl_write(path: Path, rows: list[dict[str, Any]]):
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")


def _remove_claim_lines(context: str, ids: set[str]) -> str:
    kept=[]
    for line in context.splitlines():
        if any(line.startswith(f"[{cid};") or line.startswith(f"[{cid}]") for cid in ids):
            continue
        kept.append(line)
    return "\n".join(kept)


def _claim_ids_in_context(context: str) -> list[str]:
    import re
    return re.findall(r"^\[(C\d{2}|CX\d+);", context, flags=re.MULTILINE)


def permute_source_documents(docs: list[dict], seed: int = 42) -> tuple[list[dict], list[str], list[str]]:
    """Return one deterministic permutation of source documents and both ID orders."""
    shuffled = list(docs)
    before = [d["document_id"] for d in docs]
    random.Random(seed).shuffle(shuffled)
    after = [d["document_id"] for d in shuffled]
    if after == before:
        raise AssertionError(f"random_state={seed} did not change the source-document order")
    return shuffled, before, after


def permuted_document_context(mode: str, question: dict, shuffled_docs: list[dict], claims: list[dict]) -> str:
    """Rebuild the normal mode context after permuting only the source-document list."""
    if mode == "A":
        return full_context(shuffled_docs, claims)
    return _base_context(mode, question, shuffled_docs, claims)[0]


def _base_context(mode: str, question: dict, docs: list[dict], claims: list[dict]) -> tuple[str, list[str], list[str]]:
    package = build_contexts(question["question"], 15, docs, claims)[mode]
    return package["context"], package["document_ids"], package["claim_ids"]


def _mode_context(mode: str, question: dict, docs: list[dict], claims: list[dict]) -> tuple[str, list[str], list[str]]:
    return _base_context(mode, question, docs, claims)


def build_cases() -> list[dict[str, Any]]:
    docs, claims, questions = load_documents(), load_claims(), load_questions()
    cases=[]
    skipped_noop=[]
    # One reproducible source-document permutation is shared by every question and mode.
    shuffled_docs, document_order_before, document_order_after = permute_source_documents(docs, seed=42)

    for q in questions:
        base_by_mode = {m: _base_context(m, q, docs, claims) for m in ("A", "B", "C")}
        for test_type in ("permutation", "remove_premise", "add_contradiction"):
            if test_type == "remove_premise" and q["question_id"] not in REMOVALS:
                continue  # test 2 is defined only for multi-hop and temporal questions
            for mode in ("A", "B", "C"):
                original, doc_ids, claim_ids = base_by_mode[mode]
                mutated=original
                intervention={}
                expected=q["gold_status"] if test_type == "permutation" else None
                context_changed = None
                if test_type == "permutation":
                    # Apply the same intervention to the source corpus for all modes.
                    # A consumes full documents, so its model context changes order.
                    # B/C consume BM25-selected claims; re-running their normal builders
                    # with shuffled source documents must leave the retrieval context
                    # byte-identical. This is the intended invariance control.
                    mutated=permuted_document_context(mode, q, shuffled_docs, claims)
                    if mode in {"B", "C"}:
                        _, _, permuted_claim_ids = _base_context(mode, q, shuffled_docs, claims)
                        if set(permuted_claim_ids) != set(claim_ids):
                            raise AssertionError(f"Document permutation changed retrieval membership for {q['question_id']} {mode}")
                    context_changed = mutated != original
                    intervention={
                        "random_state":42,
                        "permuted_unit":"source documents",
                        "document_order_before":document_order_before,
                        "document_order_after":document_order_after,
                        "model_context_changed":context_changed,
                        "interpretation":(
                            "full-document order is visible to the model in mode A"
                            if mode == "A" else
                            "retrieval-derived context is invariant to source-document list order; this is a same-context replication control"
                        ),
                    }
                    expected=q["gold_status"]
                    if mode == "A" and not context_changed:
                        raise AssertionError(f"Mode A document permutation did not change context for {q['question_id']}")
                    if mode in {"B", "C"} and context_changed:
                        raise AssertionError(f"Modes B/C should be invariant to source-document list order for {q['question_id']} {mode}")
                    if set(_claim_ids_in_context(mutated)) != set(claim_ids):
                        raise AssertionError(f"Permutation changed claim membership for {q['question_id']} {mode}")
                elif test_type == "remove_premise":
                    cfg=REMOVALS[q["question_id"]]
                    mutated, dropped_ids, masked_pieces = remove_fact(original, cfg["literals"], mode, claims)
                    leftovers=[lit for lit in cfg["literals"] if lit in mutated]
                    if leftovers:
                        raise AssertionError(f"Fact not fully removed for {q['question_id']} {mode}: {leftovers}")
                    if mutated == original:
                        # The fact was not in this B/C context (not retrieved), so there is nothing to remove: no intervention, no case.
                        skipped_noop.append(f"{q['question_id']}_{mode}_remove_premise")
                        continue
                    intervention={"removed_claim_ids":sorted(dropped_ids),"masked_source_pieces":masked_pieces,
                                  "fact_literals":cfg["literals"],"reason":cfg["reason"]}
                    expected=cfg["expected"]
                else:
                    spec = CONTRADICTIONS[q["question_id"]]
                    mutated, synthetic = _add_synthetic(mode, original, spec)
                    intervention = {"synthetic_claim": synthetic, "source_type": spec["source_type"], "expected_reason": spec["reason"],
                                    "warning": "CX01/X01 are synthetic, not part of the gold dataset; marked only in this metadata, not in the model context."}
                    expected = spec["expected"]
                case={
                    "case_id":f"{q['question_id']}_{mode}_{test_type}",
                    "question_id":q["question_id"],"question_type":q["question_type"],"question":q["question"],
                    "mode":mode,"test_type":test_type,"gold_status":q["gold_status"],"expected_stress_status":expected,
                    "gold_evidence_ids":q["gold_evidence_ids"],"gold_counterevidence_ids":q["gold_counterevidence_ids"],
                    "original_context_claim_ids":claim_ids,"original_context_document_ids":doc_ids,
                    "context_changed":context_changed,
                    "mutated_context_claim_ids":_claim_ids_in_context(mutated),
                    "original_context_sha256":hashlib.sha256(original.encode()).hexdigest(),
                    "original_context":original,"intervention":intervention,
                    "mutated_context_sha256":hashlib.sha256(mutated.encode()).hexdigest(),
                    "request_nonce":request_nonce(q["question_id"], q["question"], mutated),
                    "mutated_context":mutated,
                    "actual_status":"not_run","review_note":"Expected status is a manually authored evaluation target, not an observed LLM result."
                }
                cases.append(case)
    build_cases.skipped_noop = skipped_noop
    return cases


def run_stress_tests(run_llm: bool = False, api_key: str | None = None, model: str | None = None,
                     temperature: float = 0.0) -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    cases=build_cases()
    jsonl_write(RESULTS / "stress_cases.jsonl", cases)
    from llm import selected_model_name, backend_label
    key=None  # Local model: no API credential needed.
    permutation_cases = [c for c in cases if c["test_type"] == "permutation"]
    manifest={"created_at_local":datetime.now(ZoneInfo("Europe/Berlin")).isoformat(timespec="seconds"),"n_questions":12,"skipped_noop_cases":list(getattr(build_cases,"skipped_noop",[])),"stress_types":["permutation","remove_premise","add_contradiction"],"modes":["A","B","C"],
              "prepared_cases":len(cases),"permutation_design":{"seed":42,"permuted_unit":"source documents","context_changed":{"A":sum(c.get("context_changed") is True for c in permutation_cases if c["mode"]=="A"),"B":sum(c.get("context_changed") is True for c in permutation_cases if c["mode"]=="B"),"C":sum(c.get("context_changed") is True for c in permutation_cases if c["mode"]=="C")},"same_context_controls":{"B":sum(c.get("context_changed") is False for c in permutation_cases if c["mode"]=="B"),"C":sum(c.get("context_changed") is False for c in permutation_cases if c["mode"]=="C")}},
              "model":selected_model_name(model),"temperature":temperature,
              "seed":DEFAULT_SEED if backend_label()=="ollama_json_schema" else None,
              "generation_mode":"temperature_zero_greedy" if temperature==0.0 else "sampling",
              "system_prompt_sha256":hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
              "user_prompt_template_sha256":hashlib.sha256(USER_TEMPLATE.encode("utf-8")).hexdigest(),
              "backend":backend_label(),"api_key_required":False,"status":"prepared_only"}
    if run_llm:
        run_path=RESULTS/"stress_runs.jsonl"
        run_path.write_text("",encoding="utf-8")
        valid_ids={c["claim_id"] for c in load_claims()}
        for case_index, case in enumerate(cases, start=1):
            rec={k:v for k,v in case.items() if k not in {"original_context","mutated_context"}}
            rec["context_text"]=case["mutated_context"]
            ids=set(case.get("original_context_claim_ids", []))
            if case["test_type"] == "remove_premise":
                ids -= set(case.get("intervention", {}).get("removed_claim_ids", []))
            if case["test_type"] == "add_contradiction":
                ids.add("CX01")
            # A starts with every claim ID; still remove masked IDs from A after intervention.
            if case["mode"] == "A":
                ids = set(valid_ids)
                if case["test_type"] == "remove_premise":
                    ids -= set(case.get("intervention", {}).get("removed_claim_ids", []))
                if case["test_type"] == "add_contradiction":
                    ids.add("CX01")
            rec["available_claim_ids_after_intervention"] = sorted(ids)
            try:
                raw=generate_raw_answer(
                    case["question_id"], case["question"], case["mutated_context"],
                    api_key=key, model=model, temperature=temperature, valid_claim_ids=ids,
                )
                rec["raw_response"]=raw["raw_text"]
                rec.update({k:v for k,v in raw.items() if k!="raw_text"})
                if case["mode"] == "A" and raw.get("input_truncated"):
                    raise ValueError("Mode A stress context was truncated; trial is invalid")
                try:
                    answer=parse_model_answer(raw["raw_text"],ids,expected_question_id=case["question_id"])
                except Exception as first_validation_error:
                    rec["first_validation_error"] = f"{type(first_validation_error).__name__}: {first_validation_error}"
                    rec["first_raw_response"] = raw["raw_text"]
                    raw_retry = generate_raw_answer(
                        case["question_id"], case["question"], case["mutated_context"],
                        api_key=key, model=model, temperature=temperature, valid_claim_ids=ids,
                        repair_instruction=(
                            f"Your previous JSON failed validation: {first_validation_error}. "
                            "Correct the validation issue. Use lowercase status; for both, fill both ID lists; "
                            "for insufficient, start answer with 'Недостаточно данных:' and name the missing information."
                        ),
                    )
                    if case["mode"] == "A" and raw_retry.get("input_truncated"):
                        raise ValueError("Mode A stress retry was truncated; trial is invalid")
                    rec["first_raw_response"] = rec.get("raw_response")
                    rec["raw_response"] = raw_retry["raw_text"]
                    rec["retry_metadata"] = {k:v for k,v in raw_retry.items() if k!="raw_text"}
                    rec["retry_count"] = 1
                    rec["total_tokens"] = int(rec.get("total_tokens", 0)) + int(raw_retry.get("total_tokens", 0))
                    rec["elapsed_seconds"] = round(float(rec.get("elapsed_seconds", 0)) + float(raw_retry.get("elapsed_seconds", 0)), 4)
                    try:
                        answer=parse_model_answer(raw_retry["raw_text"],ids,expected_question_id=case["question_id"])
                        rec["validation_status"]="valid_after_repair"
                        rec["native_model_output_valid"]=False
                        rec["safe_fallback_used"]=False
                    except Exception as second_validation_error:
                        answer, repair_reasons, safe_fallback = parse_model_answer_with_safe_repair(
                            raw_retry["raw_text"], ids, expected_question_id=case["question_id"]
                        )
                        rec["post_validation_repair_reasons"] = repair_reasons
                        rec["safe_fallback_used"] = safe_fallback
                        rec["native_model_output_valid"] = False
                        rec["second_validation_error"] = f"{type(second_validation_error).__name__}: {second_validation_error}"
                        rec["validation_status"] = "safe_fallback" if safe_fallback else "postprocessed_after_retry"
                rec["prediction"]=answer.model_dump(mode="json")
                rec["actual_status"]=answer.status.value
                rec["matches_expected_stress_status"]=(answer.status.value==case["expected_stress_status"])
                if rec.get("validation_status") not in {"safe_fallback", "valid_after_repair", "postprocessed_after_retry"}:
                    rec["validation_status"]="valid"
                rec.setdefault("native_model_output_valid", True)
                rec.setdefault("safe_fallback_used", False)
            except Exception as exc:
                rec["validation_status"]="error"
                rec["error"]=f"{type(exc).__name__}: {exc}"
            with run_path.open("a",encoding="utf-8") as f:f.write(json.dumps(rec,ensure_ascii=False)+"\n")
            if case_index % 6 == 0 or case_index == len(cases):
                print(f"Stress progress: {case_index}/{len(cases)} cases", flush=True)
        observed = [json.loads(line) for line in run_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        valid_count = sum(row.get("validation_status") in {"valid", "valid_after_repair", "postprocessed_after_retry", "safe_fallback"} and isinstance(row.get("prediction"), dict) for row in observed)
        manifest["status"] = "completed" if valid_count == len(observed) else "completed_with_errors"
        manifest["observed_runs"] = len(observed)
        manifest["usable_stress_outputs"] = valid_count
        manifest["safe_fallback_stress_outputs"] = sum(row.get("validation_status") == "safe_fallback" for row in observed)
        manifest["native_model_valid_stress_outputs"] = sum(row.get("validation_status") == "valid" for row in observed)
        manifest["failed_or_invalid"] = len(observed) - valid_count
        manifest["results_file"] = str(run_path.relative_to(ROOT))
    (RESULTS/"stress_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return manifest

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-llm",action="store_true",help="Run all 84 stress cases with a local model; no API key required.")
    parser.add_argument("--model",default=None)
    parser.add_argument("--temperature",type=float,default=0.0)
    args=parser.parse_args()
    print(json.dumps(run_stress_tests(args.run_llm,model=args.model,temperature=args.temperature),ensure_ascii=False,indent=2))
