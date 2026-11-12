"""AI detective lab #6: interactive research stand.

This is the ORIGINAL experimental stand, using the real context/intervention
algorithms and the real Qwen runner. The recorded experiments are read-only and
shown only when the displayed context is exactly the experiment input.

UI design goals: small reruns, interventions committed via a form, no huge text or
84-row archive rendered on every widget change, and actionable errors.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app_helpers import claim_rows, comparison_rows, context_claim_ids
from common import load_documents, load_claims, load_questions
from context_builder import enrich_claims
from llm import SYSTEM_PROMPT, USER_TEMPLATE, generate_raw_answer, request_nonce, selected_model_name
from schemas import parse_model_answer, parse_model_answer_with_safe_repair
from stand_logic import (
    STRESS_NAMES, apply_intervention, base_packs, load_archives,
    matched_result, sha256_context,
)

st.set_page_config(page_title="AI-детектив — лабораторная №6", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner=False)
def archives():
    # Immutable read-only records, not 2–5 MB of JSONL copied on every rerun.
    return load_archives(ROOT)


@st.cache_resource(show_spinner=False)
def source_dossier():
    return load_documents(), load_claims(), load_questions()


@st.cache_data(show_spinner=False, max_entries=24)
def generated_packs(qid: str, question_text: str, docs: list, source_claims: list):
    # Fresh retrieval (actual BM25 and p-adic grouping) for an uploaded dossier.
    return base_packs({"question_id": qid, "question": question_text}, docs, source_claims,
                      use_recorded_context=False)


@st.cache_data(show_spinner=False, max_entries=64)
def cached_intervention(q: dict, mode: str, source_context: str, docs: list, claims: list,
                        kind: str, use_prepared: bool, claim_id: str | None, markers: str):
    # The actual context transformation still executes; reuse its result only
    # when inputs are exactly identical (prevent needless repeated BM25 work).
    return apply_intervention(q, mode, source_context, docs, claims, kind,
                              prepared_removal=use_prepared,
                              selected_claim_id=claim_id, custom_literals=markers)


def decode_uploaded(upload, label: str):
    try:
        text = upload.getvalue().decode("utf-8-sig")
        parsed = [json.loads(line) for line in text.splitlines() if line.strip()]
        if not parsed or not all(isinstance(row, dict) for row in parsed):
            raise ValueError("файл не содержит JSONL-объектов")
        return parsed
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Невозможно прочитать {label}: {exc}") from exc


def render_prediction(run: dict, claims: list[dict], q: dict, *, archived: bool = True):
    pred = run.get("prediction") or {}
    if not isinstance(pred, dict):
        st.error("В записи отсутствует корректный JSON-ответ модели")
        return
    a, b, c = st.columns(3)
    a.metric("Статус", pred.get("status", "—"))
    b.metric("Confidence", f"{pred['confidence']:.2f}" if isinstance(pred.get("confidence"), (float, int)) else "—")
    c.metric("Валидация", run.get("validation_status", "—"))
    if run.get("expected_stress_status"):
        st.caption(f"Эталон после вмешательства: {run['expected_stress_status']}")
    st.write("**Ответ:**", pred.get("answer", "—"))
    st.write("**Обоснование:**", pred.get("explanation", "—"))
    ev = pred.get("evidence_ids") or []
    counter = pred.get("counterevidence_ids") or []
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Evidence IDs:**", ", ".join(ev) if ev else "—")
    with col2:
        st.write("**Counterevidence IDs:**", ", ".join(counter) if counter else "—")
    with st.expander("Посмотреть тексты доказательств и контрдоказательств"):
        all_ids = list(dict.fromkeys(ev + counter))
        if all_ids:
            st.dataframe(claim_rows(all_ids, claims, q.get("gold_evidence_ids", []),
                                    q.get("gold_counterevidence_ids", [])),
                         hide_index=True, width="stretch")
        else:
            st.caption("Модель не указала доказательств")
    st.caption(f"Токенов: {run.get('total_tokens', '—')} · Время: {run.get('elapsed_seconds', '—')} с"
               + (" · Сохранённый эксперимент" if archived else " · Новый локальный прогон"))


def render_padic(context: str):
    st.subheader("p-adic ветви контекста")
    branches = []
    current_branch = None
    for line in context.splitlines():
        if line.startswith("ВЕТВЬ "):
            current_branch = line.removesuffix(":")
        elif line.startswith("[C") and "; source=" in line:
            cid = line.split(";", 1)[0].removeprefix("[")
            source = line.split("; source=", 1)[1].split(";", 1)[0]
            status = line.split("; status=", 1)[1].split("]", 1)[0]
            branches.append({"p-adic ветвь": current_branch or "—", "Claim ID": cid,
                             "Источник": source, "Статус": status})
    if branches:
        st.dataframe(branches, hide_index=True, width="stretch")
    else:
        st.warning("В изменённом контексте нет p-adic утверждений")


def show_comparison(a: dict, question: dict):
    records = [a["base_runs"][(question["question_id"], mid)] for mid in "ABC"
               if (question["question_id"], mid) in a["base_runs"]]
    if records:
        st.dataframe(comparison_rows(records, question), hide_index=True, width="stretch")
    else:
        st.warning("Не найдены основные результаты A/B/C")


def show_archive(a: dict, question: dict):
    st.header("Сравнение экспериментов и архив")
    st.subheader(f"Сравнение режимов A/B/C — {question['question_id']}")
    show_comparison(a, question)
    st.caption("Это исходные сохранённые прогоны. Статусы, confidence, F1, токены и время взяты из реального эксперимента.")
    met = a.get("metrics") or {}
    with st.expander("Общие метрики 36 основных прогонов"):
        table = []
        for mid in "ABC":
            m = met.get("per_mode", {}).get(mid, {})
            table.append({"Режим": mid, "Accuracy": m.get("status_accuracy"),
                          "Evidence F1": m.get("evidence_f1_macro"),
                          "Counter recall": m.get("counterevidence_recall_macro"),
                          "Both recall": m.get("both_recall"),
                          "Insufficient accuracy": m.get("insufficient_class_accuracy"),
                          "Средние токены": m.get("mean_total_tokens"),
                          "Среднее время, с": m.get("mean_elapsed_seconds")})
        st.dataframe(table, hide_index=True, width="stretch")

    st.subheader("Архив 84 стресс-прогонов")
    st.caption(f"Сохранено {len(a['base_list'])}/36 основных и {len(a['stress_list'])}/84 стрессовых ответов")
    if st.checkbox("Открыть таблицу стресс-тестов", key="archive_table_open"):
        type_choice = st.selectbox("Тип вмешательства", ["Все", "permutation", "remove_premise", "add_contradiction"], key="archive_type")
        q_choice = st.selectbox("Вопрос", ["Все"] + sorted({r["question_id"] for r in a["stress_list"]}), key="archive_q")
        m_choice = st.selectbox("Режим", ["Все", "A", "B", "C"], key="archive_mode")
        subset = [r for r in a["stress_list"] if
                  (type_choice == "Все" or r.get("test_type") == type_choice)
                  and (q_choice == "Все" or r.get("question_id") == q_choice)
                  and (m_choice == "Все" or r.get("mode") == m_choice)]
        rows = []
        for r in subset:
            p = r.get("prediction") or {}
            rows.append({"Кейс": r.get("case_id"), "Тип": r.get("test_type"),
                         "Режим": r.get("mode"), "Ожидается": r.get("expected_stress_status"),
                         "Статус": p.get("status"), "Confidence": p.get("confidence"),
                         "Верно": p.get("status") == r.get("expected_stress_status")})
        st.dataframe(rows, hide_index=True, width="stretch")
        if subset:
            selected = st.selectbox("Развернуть конкретный прогон", [r["case_id"] for r in subset], key="archive_detail")
            row = a["stress_runs"][selected]
            detail_question = next((q for q in source_dossier()[2] if q["question_id"] == row["question_id"]), question)
            render_prediction(row, source_dossier()[1], detail_question, archived=True)
            if st.checkbox("Показать контекст выбранного архивного прогона", key="show_archive_context"):
                st.code(row.get("context_text", ""), language="text")


def run_model(question: dict, mode: str, intervention_result: dict, known_claims: list[dict]):
    """Optional live experiment; never called automatically or by changing widgets."""
    context = intervention_result["context"]
    with st.spinner("Запрос к локальной модели Ollama…"):
        try:
            raw = generate_raw_answer(question["question_id"], question["question"], context,
                                      api_key=None, model=selected_model_name(), temperature=0.0)
            allowed = set(context_claim_ids(context))
            try:
                parsed = parse_model_answer(raw["raw_text"], allowed,
                                            expected_question_id=question["question_id"])
                validation_status = "valid"
            except Exception:
                parsed, _, safe_fallback = parse_model_answer_with_safe_repair(
                    raw["raw_text"], allowed, expected_question_id=question["question_id"])
                validation_status = "safe_fallback" if safe_fallback else "valid_after_repair"
            run = {"question_id": question["question_id"], "mode": mode,
                   "context_sha256": intervention_result["sha256"],
                   "context_text": context,
                   "prediction": parsed.model_dump(mode="json"),
                   "validation_status": validation_status,
                   "raw_response": raw.get("raw_text"),
                   "model": raw.get("model"), "elapsed_seconds": raw.get("elapsed_seconds"),
                   "prompt_tokens": raw.get("prompt_tokens"), "completion_tokens": raw.get("completion_tokens"),
                   "total_tokens": raw.get("total_tokens"),
                   "request_nonce": request_nonce(question["question_id"], question["question"], context),
                   "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
                   "user_prompt_template_sha256": hashlib.sha256(USER_TEMPLATE.encode()).hexdigest(),
                   "intervention": intervention_result["kind"],
                   "source": "local_manual_run"}
            st.session_state["new_model_run"] = run
            st.success("Модель ответила на текущий изменённый контекст")
        except Exception as exc:
            st.error(f"Локальная Ollama не ответила: {type(exc).__name__}: {exc}")
            st.info("Сохранённые результаты доступны без Ollama. Для нового прогона запусти Ollama и установи модель Qwen2.5-7B-Instruct.")


def stand():
    st.title("🔎 AI-детектив: расследование блокировки в «Контуре»")
    st.caption("Лабораторная №6 · один Qwen2.5-7B-Instruct · A/B/C · реальные вмешательства · проверяемые доказательства")

    try:
        a = archives()
        default_docs, default_claims, questions = source_dossier()
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        st.error(f"Ошибка загрузки файлов лабораторной: {exc}")
        st.stop()

    with st.sidebar:
        st.header("Досье")
        docs_file = st.file_uploader("Загрузить documents.jsonl", type=["jsonl", "txt"], key="upload_docs")
        claims_file = st.file_uploader("Загрузить claims.jsonl", type=["jsonl", "txt"], key="upload_claims")
        st.caption("Без загрузок используются 10 документов и 40 исходных утверждений.")
        section = st.radio("Экран", ["Исследовательский стенд", "Сравнение и архив"], key="section")

    original_dossier = docs_file is None and claims_file is None
    try:
        docs = decode_uploaded(docs_file, "documents.jsonl") if docs_file else default_docs
        claims = decode_uploaded(claims_file, "claims.jsonl") if claims_file else default_claims
        doc_ids = {d["document_id"] for d in docs}
        if len(doc_ids) != len(docs):
            raise ValueError("В досье встречаются повторяющиеся document_id")
        if not all({"document_id", "title", "source_type", "created_at", "author", "text"} <= set(d) for d in docs):
            raise ValueError("Документы должны содержать document_id, title, source_type, created_at, author, text")
        if not all({"claim_id", "document_id", "status", "text", "evidence"} <= set(c) for c in claims):
            raise ValueError("У утверждений отсутствуют обязательные поля")
        if any(c["document_id"] not in doc_ids for c in claims):
            raise ValueError("Некоторые claims ссылаются на отсутствующие document_id. Загрузи согласованный claims.jsonl")
    except (KeyError, TypeError, ValueError) as exc:
        st.error(f"Досье не загружено: {exc}")
        st.stop()

    q_labels = {f'{q["question_id"]} · {q["question"]}': q for q in questions}
    selected_q = st.selectbox("Вопрос", list(q_labels), key="question_select")
    q = q_labels[selected_q]
    mode_label = st.radio("Режим", ["A — полный контекст", "B — BM25 top-15", "C — p-adic контекст"],
                          horizontal=True, key="mode_select")
    mode = mode_label[0]

    if section == "Сравнение и архив":
        if not original_dossier:
            st.warning("Архивные ответы получены для исходного досье, а не загруженных файлов")
        show_archive(a, q)
        return

    try:
        packs = (base_packs(q, docs, claims, a, use_recorded_context=True)
                 if original_dossier else generated_packs(q["question_id"], q["question"], docs, claims))
        pack = packs[mode]
    except (ValueError, KeyError, AssertionError) as exc:
        st.error(f"Не удалось построить контекст: {exc}")
        st.stop()

    st.subheader("Вмешательства в досье")
    st.caption("Выбор варианта сам по себе не запускает модель. Нажми «Применить», чтобы получить изменённый контекст настоящим алгоритмом эксперимента.")
    kinds = list(STRESS_NAMES)
    # A form prevents transient reruns at every change of a control.
    with st.form("intervention_form", clear_on_submit=False):
        kind_choice = st.selectbox("Вариант вмешательства", kinds,
                                   format_func=lambda k: STRESS_NAMES[k], key="intervention_kind")
        all_claim_ids = [cid for cid in pack["claim_ids"] if cid in {c["claim_id"] for c in claims}]
        choices = ["Подготовленная посылка из эксперимента (Q03–Q06)"] + all_claim_ids
        claim_choice = st.selectbox("Посылка для удаления (если выбран этот вариант)", choices,
                                    key="selected_removal_claim")
        markers = st.text_input("Свои маркеры удаляемого факта — через ; (необязательно)",
                                key="removal_markers")
        submitted = st.form_submit_button("Применить вмешательство", type="primary")
    if submitted:
        st.session_state["applied_intervention"] = {
            "kind": kind_choice, "selected": claim_choice, "markers": markers,
            "question_id": q["question_id"], "mode": mode,
        }
    chosen = st.session_state.get("applied_intervention", {})
    # On Q/mode change, reset the applied intervention instead of using a
    # half-applicable selected premise from the previous question/mode.
    if chosen.get("question_id") != q["question_id"] or chosen.get("mode") != mode:
        chosen = {"kind": "none"}

    try:
        removal_choice = chosen.get("selected", "")
        manual_claim = removal_choice if removal_choice.startswith("C") else None
        using_prepared = not manual_claim
        result = cached_intervention(
            q, mode, pack["context"], docs, claims, chosen.get("kind", "none"),
            using_prepared, manual_claim, chosen.get("markers", ""),
        )
    except (KeyError, ValueError, AssertionError) as exc:
        st.error(f"Вмешательство не применено: {type(exc).__name__}: {exc}")
        st.info("Вернись к варианту «Без вмешательства» или выбери другую посылку")
        result = apply_intervention(q, mode, pack["context"], docs, claims, "none")

    st.write("**Итог:**", STRESS_NAMES[result["kind"]])
    st.caption(result["note"])
    if result["kind"] != "none":
        st.success("Контекст изменён" if result["context_changed"] else "Контекст не изменился — контроль инвариантности")
    st.caption(f"SHA-256 контекста: `{result['sha256']}`")
    st.info(f"Документов в досье: {len(docs)} · Утверждений: {len(claims)} · Claim IDs в текущем контексте: {len(result['claim_ids'])}")

    if mode in {"B", "C"}:
        b_ids, c_ids = set(packs["B"]["claim_ids"]), set(packs["C"]["claim_ids"])
        st.caption(f"B/C top-15: одинаковые ID = {b_ids == c_ids} · {len(b_ids)} утверждений")
    if mode == "C":
        render_padic(result["context"])

    with st.expander("Посмотреть или скачать изменённый контекст"):
        if st.checkbox("Показать полный текст контекста", key="expand_context_text"):
            st.text_area("Контекст после вмешательства", result["context"], height=320, disabled=True)
        st.download_button("Скачать контекст TXT", result["context"].encode("utf-8"),
                           file_name=f'{q["question_id"]}_{mode}_{result["kind"]}.txt', mime="text/plain")

    st.subheader("Ответ и confidence")
    match, message = matched_result(a, q["question_id"], mode, result, original_dossier)
    if match:
        st.caption(message)
        known_claims = claims + ([result["synthetic_claim"]] if result["synthetic_claim"] else [])
        render_prediction(match, known_claims, q)
    else:
        st.warning(message)

    st.write("**Новый ответ модели:**")
    st.caption("В отличие от архивного просмотра, кнопка действительно отправляет изменённый контекст в локальную Ollama. Нужны запущенная Ollama и Qwen2.5-7B-Instruct.")
    if st.button("Отправить текущий контекст в Ollama", key="run_model", type="secondary"):
        run_model(q, mode, result, claims)
    previous = st.session_state.get("new_model_run")
    if previous and previous.get("question_id") == q["question_id"] and previous.get("mode") == mode and previous.get("context_sha256") == result["sha256"]:
        render_prediction(previous, claims + ([result["synthetic_claim"]] if result["synthetic_claim"] else []), q, archived=False)
        if st.button("Сохранить новый прогон в results/manual_runs.jsonl"):
            dest = ROOT / "results" / "manual_runs.jsonl"
            with dest.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(previous, ensure_ascii=False) + "\n")
            st.success("Новый прогон сохранён отдельно от основного эксперимента")

    st.subheader("Сравнение сохранённых результатов A/B/C")
    if original_dossier:
        show_comparison(a, q)
    else:
        st.warning("Это пользовательское досье: исходное сравнение A/B/C было получено на другом наборе данных")
    with st.expander("Эталон для проверки ответа"):
        st.write("Статус:", q["gold_status"])
        st.write("Ответ:", q["gold_answer"])
        st.write("Evidence IDs:", q["gold_evidence_ids"])
        st.write("Counterevidence IDs:", q["gold_counterevidence_ids"])


stand()
