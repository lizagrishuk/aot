from __future__ import annotations
import sys, json, hashlib, os, urllib.request, urllib.error
from pathlib import Path
from collections import defaultdict
import streamlit as st

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from common import load_documents, load_claims, load_questions
from context_builder import build_contexts
from llm import generate_raw_answer, selected_model_name, SYSTEM_PROMPT, USER_TEMPLATE, request_nonce
from schemas import parse_model_answer, parse_model_answer_with_safe_repair
from app_helpers import claim_rows, removal_literals, comparison_rows, read_jsonl, context_claim_ids, context_document_ids

st.set_page_config(page_title="AI-детектив", page_icon="🔎", layout="wide")
st.title("🔎 AI-детектив: расследование блокировки в «Контуре»")
st.caption("Вымышленный кейс. Используется одна локальная модель Qwen2.5-7B-Instruct (Ollama); API-ключ не нужен. При первом запуске скачиваются публичные веса.")

# Optional uploads: source documents and matching claims. Uploaded content is kept only in this session.
with st.sidebar:
    st.header("Данные")
    docs_upload=st.file_uploader("Загрузить documents.jsonl",type=["jsonl","txt"],key="docs_upload")
    claims_upload=st.file_uploader("Загрузить claims.jsonl (необязательно)",type=["jsonl","txt"],key="claims_upload")
    st.caption("Если документы используют новые ID, загрузите соответствующий claims.jsonl тоже.")

try:
    docs=[json.loads(x) for x in docs_upload.getvalue().decode("utf-8").splitlines() if x.strip()] if docs_upload else load_documents()
    claims=[json.loads(x) for x in claims_upload.getvalue().decode("utf-8").splitlines() if x.strip()] if claims_upload else load_claims()
    questions=load_questions()
    doc_ids={d["document_id"] for d in docs}
    claim_ids={c["claim_id"] for c in claims}
    required_doc_fields={"document_id","title","source_type","created_at","author","text"}
    if any(not required_doc_fields.issubset(d) for d in docs): raise ValueError("documents.jsonl is missing required fields")
    if any(c.get("document_id") not in doc_ids for c in claims): raise ValueError("A claim references a document ID that is absent from the uploaded dossier")
except Exception as exc:
    st.error(f"Не удалось загрузить данные: {exc}")
    st.stop()

label_to_q={f'{q["question_id"]} · {q["question"]}':q for q in questions}
selected_label=st.selectbox("Вопрос",list(label_to_q))
q=label_to_q[selected_label]
mode=st.radio("Режим",["A — полный контекст","B — BM25 top-15","C — p-adic контекст"],horizontal=True)
mode_id=mode[0]
try:
    packs=build_contexts(q["question"],15,docs,claims)
except Exception as exc:
    st.error(f"Ошибка построения контекста: {exc}")
    st.stop()
selected_pack=packs[mode_id]
context=selected_pack["context"]

# Stress controls modify one intervention at a time, and the mutated context is shown before a model call.
# The interventions are the same functions that prepare the 84 recorded stress cases (src/stress_tests.py).
from stress_tests import REMOVALS, CONTRADICTIONS, _add_synthetic, permute_source_documents, permuted_document_context
from interventions import remove_fact
with st.expander("Стресс-тесты для выбранного вопроса",expanded=False):
    intervention=st.selectbox("Вариант вмешательства",["Без вмешательства","Перестановка (random_state=42)","Удалить один факт","Добавить подготовленное противоречие"])
    chosen_claims=st.multiselect("Или выберите утверждения-посылки для удаления",
                                 [c["claim_id"] for c in claims],
                                 disabled=intervention!="Удалить один факт",
                                 help="Удаляются выбранные утверждения и фрагменты источников, из которых они взяты (дословное evidence). Если выбор пуст, используются маркеры ниже.")
    custom_literals=st.text_input("Маркеры факта для удаления (через ;)",value="; ".join(REMOVALS.get(q["question_id"],{}).get("literals",[])),
                                  disabled=intervention!="Удалить один факт",
                                  help="Удаляются все утверждения и все предложения источников, содержащие любой из маркеров.")
    applied_context=context
    intervention_note="Без вмешательства"
    if intervention.startswith("Перестановка"):
        shuffled, order_before, order_after = permute_source_documents(docs, seed=42)
        applied_context=permuted_document_context(mode_id,q,shuffled,claims)
        if mode_id=="A":
            intervention_note=f"Исходные документы переставлены с random_state=42: {order_before} → {order_after}. Режим A получает полный текст в новом порядке."
        else:
            if applied_context != context:
                st.error("Нарушена инвариантность retrieval: B/C изменили контекст при перестановке исходного списка документов.")
            intervention_note=f"Исходные документы переставлены с random_state=42: {order_before} → {order_after}. Retrieval-контекст B/C остаётся идентичным; это контроль устойчивости построителя к порядку входного списка."
    elif intervention=="Удалить один факт":
        literals=removal_literals(chosen_claims,claims) or [x.strip() for x in custom_literals.split(";") if x.strip()]
        if not literals:
            st.warning("Укажите хотя бы один маркер факта.")
        else:
            applied_context,dropped,masked=remove_fact(context,literals,mode_id,claims)
            intervention_note=f"Удалены утверждения {dropped or '—'}; замаскировано фрагментов источников: {masked}. Оставшиеся маркеры в контексте: {[x for x in literals if x in applied_context] or 'нет'}."
    elif intervention=="Добавить подготовленное противоречие":
        spec=CONTRADICTIONS.get(q["question_id"])
        if spec is None:
            st.warning("Для этого вопроса подготовленного противоречия нет.")
        else:
            applied_context,_=_add_synthetic(mode_id,context,spec)
            intervention_note=f"Добавлена запись CX01 типа {spec['source_type']} (в контексте без пометки о тесте). Ожидаемый статус: {spec['expected']}. {spec['reason']}"
    st.caption(intervention_note)
    st.text_area("Контекст после вмешательства (для проверки)",applied_context,height=180)
    context=applied_context

# The validator must accept only claim IDs actually present in the post-intervention context.
allowed_ids=set(context_claim_ids(context))
context_doc_ids=context_document_ids(context, selected_pack.get("document_ids", []))

st.subheader("Контекст выбранного режима")
st.info(f"Документов в досье: {len(docs)} · Утверждений в исходном наборе: {len(claims)} · Claim IDs в контексте после вмешательства: {len(allowed_ids)}")
with st.expander("Показать полный контекст, передаваемый модели"):
    st.text(context)
st.download_button("Скачать контекст",context.encode("utf-8"),file_name=f'{q["question_id"]}_{mode_id}_context.txt',mime="text/plain")
current_context_sha256=hashlib.sha256(context.encode("utf-8")).hexdigest()

if mode_id in {"B","C"}:
    st.subheader("Проверка одинакового набора утверждений B/C")
    st.write({"B_count":len(packs["B"]["claim_ids"]),"C_count":len(packs["C"]["claim_ids"]),
              "same_claim_ids":set(packs["B"]["claim_ids"])==set(packs["C"]["claim_ids"]),
              "claim_ids":packs["B"]["claim_ids"]})
if mode_id=="C":
    st.subheader("p-adic ветви")
    enriched=[]
    for c in packs["C"]["retrieval"]:
        enriched.append({"claim_id":c["claim_id"],"path":c["p_adic_path"],"code":c["p_adic_code"],"source":c["document_id"],"status":c["status"],"claim":c["text"]})
    st.dataframe(enriched,use_container_width=True,hide_index=True)

if intervention.startswith("Перестановка") and mode_id in {"B", "C"} and st.session_state.get("last_raw", {}).get("context_sha256") == current_context_sha256:
    st.info("В B/C после перестановки исходных документов prompt остаётся идентичным. Показанный выше ответ относится к тому же контексту; чтобы получить новый повторный ответ и измерить вариативность генерации, нажмите кнопку запуска ещё раз.")

st.subheader("Запуск одной модели")
model_name=selected_model_name()
st.caption(f"Единая модель: {model_name} (Ollama, temperature=0)")

def ollama_is_available() -> bool:
    url=os.getenv("OLLAMA_URL","http://127.0.0.1:11434").rstrip("/")+"/api/tags"
    try:
        with urllib.request.urlopen(url,timeout=1):
            return True
    except Exception:
        return False

ollama_available=ollama_is_available()
if not ollama_available:
    st.info("Открыт режим просмотра сохранённых результатов. Для новых запросов к Qwen запустите Ollama локально; в Streamlit Cloud доступен архивный просмотр без повторной генерации.")
if st.button("Отправить вопрос локальной модели",type="primary",disabled=not ollama_available):
    with st.spinner("Загружаю модель (при первом запуске) и выполняю запрос…"):
        st.session_state["last_validation_error"] = None
        st.session_state["last_validation_status"] = None
        st.session_state["last_repair_reasons"] = []
        try:
            raw=generate_raw_answer(q["question_id"],q["question"],context,api_key=None,model=model_name or None,temperature=0.0)
            st.session_state["last_raw"]={"question_id":q["question_id"],"question":q["question"],"mode":mode_id,"raw":raw,
                "context":context,"context_sha256":hashlib.sha256(context.encode()).hexdigest(),"context_claim_ids":sorted(allowed_ids),
                "context_document_ids":context_doc_ids,"gold_status":q["gold_status"],
                "gold_evidence_ids":q["gold_evidence_ids"],"gold_counterevidence_ids":q["gold_counterevidence_ids"],
                "system_prompt_sha256":hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
                "user_prompt_template_sha256":hashlib.sha256(USER_TEMPLATE.encode("utf-8")).hexdigest(),
                "request_nonce":request_nonce(q["question_id"],q["question"],context)}
            try:
                answer=parse_model_answer(raw["raw_text"],allowed_ids,expected_question_id=q["question_id"])
                st.session_state["last_validation_status"]="valid"
                st.session_state["last_repair_reasons"]=[]
            except Exception as strict_error:
                answer, repair_reasons, safe_fallback = parse_model_answer_with_safe_repair(
                    raw["raw_text"], allowed_ids, expected_question_id=q["question_id"]
                )
                st.session_state["last_validation_status"]="safe_fallback" if safe_fallback else "valid_after_repair"
                st.session_state["last_repair_reasons"]=repair_reasons
                st.session_state["last_validation_error"]=f"Strict validation failed: {strict_error}"
            st.session_state["last_parsed"]=answer.model_dump(mode="json")
            st.session_state.setdefault("last_validation_error",None)
        except Exception as exc:
            st.session_state["last_parsed"]=None
            st.session_state["last_validation_error"]=f"{type(exc).__name__}: {exc}"
            if st.session_state.get("last_raw") is None: st.session_state["last_raw"]={"question_id":q["question_id"],"mode":mode_id,"raw":{"raw_text":""}}
if (st.session_state.get("last_raw",{}).get("question_id")==q["question_id"]
    and st.session_state.get("last_raw",{}).get("mode")==mode_id
    and st.session_state.get("last_raw",{}).get("context_sha256")==current_context_sha256):
    raw=st.session_state["last_raw"]["raw"]
    st.write({"model":raw.get("model"),"elapsed_seconds":raw.get("elapsed_seconds"),"prompt_tokens":raw.get("prompt_tokens"),"completion_tokens":raw.get("completion_tokens"),"total_tokens":raw.get("total_tokens")})
    if st.session_state.get("last_validation_error"):
        st.warning(st.session_state["last_validation_error"])
        if st.session_state.get("last_validation_status")=="safe_fallback":
            st.error("Показан безопасный отказ системы, а не валидный нативный ответ модели. Сырой ответ сохранён для аудита.")
        if st.session_state.get("last_repair_reasons"):
            st.write("Причины исправления:", st.session_state["last_repair_reasons"])
        st.code(raw.get("raw_text",""),language="json")
    elif st.session_state.get("last_parsed"):
        a=st.session_state["last_parsed"]
        st.metric("Статус ответа",a["status"])
        st.metric("Confidence",f'{a["confidence"]:.2f}')
        st.write("**Ответ:**",a["answer"])
        st.write("**Explanation:**",a["explanation"])
        c1,c2=st.columns(2)
        with c1:st.write("**Evidence IDs**",a["evidence_ids"])
        with c2:st.write("**Counterevidence IDs**",a["counterevidence_ids"])
        with st.expander("Тексты утверждений из ответа и сверка с эталоном",expanded=True):
            st.write("Evidence (модель):");st.dataframe(claim_rows(a["evidence_ids"],claims,q["gold_evidence_ids"],q["gold_counterevidence_ids"]),hide_index=True,use_container_width=True)
            st.write("Counterevidence (модель):");st.dataframe(claim_rows(a["counterevidence_ids"],claims,q["gold_evidence_ids"],q["gold_counterevidence_ids"]),hide_index=True,use_container_width=True)
        st.download_button("Скачать JSON-ответ",json.dumps(a,ensure_ascii=False,indent=2),file_name=f'{q["question_id"]}_{mode_id}_answer.json',mime="application/json")
        if st.button("Сохранить этот прогон в results/manual_runs.jsonl"):
            run=st.session_state["last_raw"]
            out_path=ROOT/"results"/"manual_runs.jsonl"
            out_path.parent.mkdir(parents=True,exist_ok=True)
            run_id=f'manual_{run["question_id"]}_{run["mode"]}_{run["context_sha256"][:10]}'
            existing=[]
            if out_path.exists():
                for line in out_path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        try:existing.append(json.loads(line))
                        except json.JSONDecodeError:pass
            if any(x.get("run_id")==run_id for x in existing):
                st.info("Этот прогон уже сохранён.")
            else:
                rec={"run_id":run_id,"question_id":run["question_id"],"question_type":q["question_type"],"mode":run["mode"],"question":run["question"],
                    "gold_status":run["gold_status"],"gold_evidence_ids":run["gold_evidence_ids"],"gold_counterevidence_ids":run["gold_counterevidence_ids"],
                    "context_claim_ids":run["context_claim_ids"],"context_document_ids":run["context_document_ids"],"context_sha256":run["context_sha256"],"context_text":run["context"],
                    "system_prompt_sha256":run["system_prompt_sha256"],"user_prompt_template_sha256":run["user_prompt_template_sha256"],
                    "request_nonce":run["request_nonce"],"temperature":raw.get("temperature",0.0),"seed":raw.get("seed"),
                    "generation_mode":raw.get("generation_mode"),"backend":raw.get("backend"),"requested_model":raw.get("model"),"raw_response":raw.get("raw_text"),
                    "model":raw.get("model"),"elapsed_seconds":raw.get("elapsed_seconds"),"prompt_tokens":raw.get("prompt_tokens"),"completion_tokens":raw.get("completion_tokens"),"total_tokens":raw.get("total_tokens"),
                    "prediction":a,"validation_status":st.session_state.get("last_validation_status","valid"),
                    "native_model_output_valid":st.session_state.get("last_validation_status","valid")=="valid",
                    "safe_fallback_used":st.session_state.get("last_validation_status")=="safe_fallback",
                    "post_validation_repair_reasons":st.session_state.get("last_repair_reasons",[]),"source":"streamlit_manual_run","intervention":intervention_note}
                with out_path.open("a",encoding="utf-8") as f:f.write(json.dumps(rec,ensure_ascii=False)+"\n")
                st.success("Ответ и контекст сохранены в results/manual_runs.jsonl (основные результаты эксперимента не затронуты).")
elif (st.session_state.get("last_raw",{}).get("question_id")==q["question_id"]
      and st.session_state.get("last_raw",{}).get("mode")==mode_id):
    st.info("Контекст изменился после последнего запуска. Предыдущий ответ скрыт; отправьте новый контекст модели.")

st.divider()
st.subheader("Сравнение сохранённых прогонов A/B/C")
base_runs=read_jsonl(ROOT/"results"/"llm_runs.jsonl")
manual_runs=read_jsonl(ROOT/"results"/"manual_runs.jsonl")
if base_runs or manual_runs:
    st.caption(f"Выбранный вопрос {q['question_id']} (эталон: {q['gold_status']}). Берётся последний сохранённый прогон каждого режима; ручные прогоны из стенда помечены в колонке «Источник».")
    st.dataframe(comparison_rows(base_runs+manual_runs,q),use_container_width=True,hide_index=True)
    with st.expander("Все сохранённые прогоны эксперимента"):
        st.dataframe([{"question_id":r.get("question_id"),"mode":r.get("mode"),"status":(r.get("prediction") or {}).get("status","—"),
                       "confidence":(r.get("prediction") or {}).get("confidence"),"tokens":r.get("total_tokens"),"seconds":r.get("elapsed_seconds"),
                       "validation":r.get("validation_status")} for r in base_runs],use_container_width=True,hide_index=True)
else:
    st.info("Сохранённых прогонов нет: выполните эксперимент (`python src/experiment.py`) или сохраните прогон из стенда.")

with st.expander("Эталон ответа (только после попытки ответить)"):
    st.warning("Не используйте этот эталон в контексте модели при оценке.")
    st.write("**Gold status:**",q["gold_status"])
    st.write("**Gold answer:**",q["gold_answer"])
    st.write("**Evidence IDs:**",q["gold_evidence_ids"])
    st.write("**Counterevidence IDs:**",q["gold_counterevidence_ids"])


# Read-only archive viewer: this works on Streamlit Cloud without a local Ollama server.
st.divider()
st.header("Архив результатов финального эксперимента")
st.caption("36 основных и 84 стрессовых наблюдения из сохранённого прогона. Просмотр не вызывает модель и не изменяет результаты.")

def _load_json_file(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except (OSError, json.JSONDecodeError):
        return None

archived_base=read_jsonl(ROOT/"results"/"llm_runs.jsonl")
archived_stress=read_jsonl(ROOT/"results"/"stress_runs.jsonl")
archived_lm=_load_json_file(ROOT/"results"/"llm_metrics.json")
archived_sm=_load_json_file(ROOT/"results"/"stress_metrics.json")
archived_rm=_load_json_file(ROOT/"results"/"retrieval_metrics.json")

if archived_base and archived_stress:
    m1,m2,m3,m4=st.columns(4)
    m1.metric("Основные ответы",f"{len(archived_base)}/36")
    m2.metric("Стресс-ответы",f"{len(archived_stress)}/84")
    m3.metric("BM25 accuracy",f"{(archived_lm or {}).get('per_mode',{}).get('B',{}).get('status_accuracy',0)*100:.1f}%")
    m4.metric("B/C одинаковый набор", "Да" if (archived_rm or {}).get("B_C_same_claim_ids_all_questions") else "Нет")

    with st.expander("Метрики основного эксперимента",expanded=True):
        if archived_lm:
            base_metric_rows=[]
            for mid in ("A","B","C"):
                mm=archived_lm.get("per_mode",{}).get(mid,{})
                base_metric_rows.append({
                    "Режим":mid,
                    "Accuracy":mm.get("status_accuracy"),
                    "Evidence F1":mm.get("evidence_f1_macro"),
                    "Counter recall":mm.get("counterevidence_recall_macro"),
                    "Both recall":mm.get("both_recall"),
                    "Insufficient accuracy":mm.get("insufficient_class_accuracy"),
                    "Токены":mm.get("mean_total_tokens"),
                    "Время, с":mm.get("mean_elapsed_seconds")
                })
            st.dataframe(base_metric_rows,use_container_width=True,hide_index=True)
            st.caption(f"Валидные результаты: {archived_lm.get('valid_runs',0)}/36; safe_fallback: {archived_lm.get('safe_fallback_runs',0)}.")
    with st.expander("Все 36 основных ответов"):
        base_table=[]
        for r in archived_base:
            p=r.get("prediction") or {}
            base_table.append({
                "Вопрос":r.get("question_id"),"Режим":r.get("mode"),
                "Эталон":r.get("gold_status"),"Ответ":p.get("status"),
                "Верно":p.get("status")==r.get("gold_status"),
                "Confidence":p.get("confidence"),"Evidence":", ".join(p.get("evidence_ids",[])),
                "Counterevidence":", ".join(p.get("counterevidence_ids",[])),
                "Валидация":r.get("validation_status"),"Время, с":r.get("elapsed_seconds")
            })
        st.dataframe(base_table,use_container_width=True,hide_index=True)
    with st.expander("Стресс-метрики",expanded=True):
        if archived_sm:
            stress_metric_rows=[]
            for test_name,vals in archived_sm.get("by_test",{}).items():
                stress_metric_rows.append({"Вмешательство":test_name,**vals})
            st.dataframe(stress_metric_rows,use_container_width=True,hide_index=True)
            st.caption(f"Валидные результаты с учётом повторной попытки: {archived_sm.get('model_valid_stress_runs',0)}/84; safe_fallback: {archived_sm.get('safe_fallback_outputs',0)}.")
    with st.expander("Все 84 стресс-ответа",expanded=True):
        test_names=["Все"]+sorted({r.get("test_type","") for r in archived_stress})
        chosen_test=st.selectbox("Тип стресс-теста",test_names,key="archive_stress_type")
        q_names=["Все"]+sorted({r.get("question_id","") for r in archived_stress})
        chosen_q=st.selectbox("Вопрос",q_names,key="archive_stress_question")
        mode_names=["Все","A","B","C"]
        chosen_mode=st.selectbox("Режим",mode_names,key="archive_stress_mode")
        filtered=[r for r in archived_stress
                  if (chosen_test=="Все" or r.get("test_type")==chosen_test)
                  and (chosen_q=="Все" or r.get("question_id")==chosen_q)
                  and (chosen_mode=="Все" or r.get("mode")==chosen_mode)]
        stress_table=[]
        for r in filtered:
            p=r.get("prediction") or {}
            stress_table.append({
                "ID":r.get("case_id"),"Тип":r.get("test_type"),"Вопрос":r.get("question_id"),
                "Режим":r.get("mode"),"Ожидание":r.get("expected_stress_status"),
                "Ответ":p.get("status"),"Совпадение":p.get("status")==r.get("expected_stress_status"),
                "Confidence":p.get("confidence"),"Валидация":r.get("validation_status")
            })
        st.dataframe(stress_table,use_container_width=True,hide_index=True)
        if filtered:
            selected_case=st.selectbox("Подробности стресс-кейса",
                [r["case_id"] for r in filtered],key="archive_stress_detail")
            detail=next(r for r in filtered if r["case_id"]==selected_case)
            prediction=detail.get("prediction") or {}
            d1,d2,d3=st.columns(3)
            d1.metric("Эталон",detail.get("expected_stress_status","—"))
            d2.metric("Ответ модели",prediction.get("status","—"))
            d3.metric("Валидация",detail.get("validation_status","—"))
            st.write("**Вопрос:**",detail.get("question",""))
            st.write("**Вмешательство:**",detail.get("intervention",{}))
            st.write("**Ответ:**",prediction.get("answer",""))
            st.write("**Обоснование:**",prediction.get("explanation",""))
            st.write("**Evidence IDs:**",prediction.get("evidence_ids",[]))
            st.write("**Counterevidence IDs:**",prediction.get("counterevidence_ids",[]))
            with st.expander("Контекст после вмешательства"):
                st.text(detail.get("context_text",""))
else:
    st.warning("Архивные результаты не найдены. Для демонстрации загрузите в репозиторий папку results из итогового lab6_submission.zip.")
