#!/usr/bin/env python3
"""Fail-closed final export: 36 main + 84 stress observations, report <=1500 words, PDF and ZIP."""
from __future__ import annotations
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
MODEL = 'qwen2.5:7b-instruct'
VALID = {'valid', 'valid_after_repair', 'postprocessed_after_retry', 'safe_fallback'}
STRICT_MODEL = {'valid', 'valid_after_repair'}


def load_json(path):
    if not path.exists(): raise RuntimeError(f'Нет обязательного файла: {path}')
    return json.loads(path.read_text(encoding='utf-8'))


def load_jsonl(path):
    if not path.exists(): raise RuntimeError(f'Нет обязательного файла: {path}')
    return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]


def metric(x, decimals=3):
    return 'не измерено' if x is None else f'{x:.{decimals}f}'.replace('.', ',')


def percent(x):
    return 'не измерено' if x is None else f'{x * 100:.1f}%'.replace('.', ',')


def validate():
    base = load_jsonl(RESULTS/'llm_runs.jsonl')
    stress = load_jsonl(RESULTS/'stress_runs.jsonl')
    case_rows = load_jsonl(RESULTS/'stress_cases.jsonl')
    base_manifest = load_json(RESULTS/'experiment_manifest.json')
    stress_manifest = load_json(RESULTS/'stress_manifest.json')
    lm = load_json(RESULTS/'llm_metrics.json')
    sm = load_json(RESULTS/'stress_metrics.json')
    rm = load_json(RESULTS/'retrieval_metrics.json')
    questions = load_jsonl(ROOT/'data/questions.jsonl')
    claims = load_jsonl(ROOT/'data/claims.jsonl')
    docs = load_jsonl(ROOT/'data/documents.jsonl')
    if len(questions) != 12 or len(claims) not in range(30,41) or len(docs) not in range(8,11):
        raise RuntimeError('Объем данных не соответствует лабораторной')
    if len(base) != 36 or len(stress) != 84 or len(case_rows) != 84:
        raise RuntimeError(f'Неполный эксперимент: base={len(base)}/36, stress={len(stress)}/84, cases={len(case_rows)}/84')
    if len({(r['question_id'],r['mode']) for r in base}) != 36:
        raise RuntimeError('Повторы или пропуски ключей основного эксперимента')
    if len({(r['question_id'],r['mode'],r['test_type']) for r in stress}) != 84:
        raise RuntimeError('Повторы или пропуски ключей стресс-прогона')
    counts = Counter(r['test_type'] for r in stress)
    if counts != {'permutation':36, 'remove_premise':12, 'add_contradiction':36}:
        raise RuntimeError(f'Неправильное распределение стрессов: {counts}')
    if any(r.get('validation_status') not in VALID or not isinstance(r.get('prediction'), dict) for r in base+stress):
        raise RuntimeError('Есть невалидные результаты: сохранены в result files, но отчет не создаётся')
    if any(r.get('input_truncated') for r in base+stress if r['mode'] == 'A'):
        raise RuntimeError('Полный контекст A усечён: результаты недействительны')
    if any(r.get('model') != MODEL for r in base+stress):
        raise RuntimeError('В эксперименте используется не одна фиксированная модель: '+repr(Counter(r.get('model') for r in base+stress)))
    if base_manifest.get('model') != MODEL or stress_manifest.get('model') != MODEL:
        raise RuntimeError('В манифестах записана другая модель')
    if stress_manifest.get('observed_runs') != 84 or sm.get('actual_stress_runs') != 84:
        raise RuntimeError('Стресс-метрики построены не по 84 наблюдаемым ответам модели')
    if lm.get('total_saved_runs') != 36 or not rm.get('B_C_same_claim_ids_all_questions'):
        raise RuntimeError('Основные метрики/справедливость B/C не подтверждены')
    if base_manifest.get('system_prompt_sha256') != stress_manifest.get('system_prompt_sha256'):
        raise RuntimeError('Промпт между A/B/C и стресс-тестами различается')
    if base_manifest.get('user_prompt_template_sha256') != stress_manifest.get('user_prompt_template_sha256'):
        raise RuntimeError('Шаблон вопроса отличается между режимами')
    for r in base+stress:
        if r.get('temperature') != 0.0: raise RuntimeError('Непостоянная температура')
    return dict(base=base,stress=stress,lm=lm,sm=sm,rm=rm,docs=docs,questions=questions,
                base_manifest=base_manifest,stress_manifest=stress_manifest)


def create_report(data):
    lm,sm,rm = data['lm'],data['sm'],data['rm']
    a,b,c = (lm['per_mode'][m] for m in 'ABC')
    pe = sm['by_test']['permutation']
    remove = sm['by_test']['remove_premise']
    add = sm['by_test']['add_contradiction']
    n_model = sum(r['validation_status'] in STRICT_MODEL for r in data['base'])
    n_fall = sum(r['validation_status'] == 'safe_fallback' for r in data['base'])
    n_repaired = sum(r['validation_status'] == 'valid_after_repair' for r in data['base'])
    n_stress_model = sum(r['validation_status'] in STRICT_MODEL for r in data['stress'])
    n_stress_fall = sum(r['validation_status'] == 'safe_fallback' for r in data['stress'])
    n_words_dossier = len(re.findall(r'\w+', ' '.join(d['text'] for d in data['docs'])))
    statuses = Counter(q['gold_status'] for q in data['questions'])
    p_better = (b['status_accuracy'] or 0) > (c['status_accuracy'] or 0)
    comp = 'выше' if p_better else 'ниже или равна'
    report=f'''# Отчёт: лабораторная работа № 6 «AI-детектив»

## Цель, корпус и организация

Цель — проверить доказательные ответы одной языковой модели на вопросы к вымышленному досье о блокировке пользователя в социальной сети «Контур». Созданы {len(data['docs'])} документов ({n_words_dossier} слов при подсчёте по регулярному выражению), 40 атомарных утверждений с дословными цитатами и 12 вопросов: по два прямых, multi-hop, временных, «правило + исключение», противоречия и недостаточности данных. Корпус содержит свидетельства, переписку, регламент, исключение, системный журнал, конфликт источников и два нерелевантных документа. Эталонные статусы: {dict(statuses)}. Истинная хронология и обоснования приведены в `ground_truth.md` и `data/questions.jsonl`.

Использовалась **одна модель** `qwen2.5:7b-instruct` через Ollama, `temperature=0`, `seed=42`, контекст 32768 токенов и единый системный промпт. **A:** все документы и компактный реестр `claim_id`; **B:** BM25 top-15 утверждений плоским списком; **C:** те же 15 `claim_id`, сгруппированные по p-adic ветвям и внутри каждой ветви отсортированные по времени. Совпадение составов B/C проверено для всех 12 вопросов. Базовый эксперимент: 36 запросов; стресс-эксперимент: 84 запроса. Модель не получает эталонный ответ в контексте.

## Результаты основного эксперимента

| Режим | Статус accuracy* | Evidence F1* | Counter recall* | Both recall* | Insufficient accuracy* | Токены* | Время, с* |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | {percent(a['status_accuracy'])} | {metric(a['evidence_f1_macro'])} | {metric(a['counterevidence_recall_macro'])} | {metric(a['both_recall'])} | {metric(a['insufficient_class_accuracy'])} | {metric(a['mean_total_tokens'],0)} | {metric(a['mean_elapsed_seconds'],2)} |
| B | {percent(b['status_accuracy'])} | {metric(b['evidence_f1_macro'])} | {metric(b['counterevidence_recall_macro'])} | {metric(b['both_recall'])} | {metric(b['insufficient_class_accuracy'])} | {metric(b['mean_total_tokens'],0)} | {metric(b['mean_elapsed_seconds'],2)} |
| C | {percent(c['status_accuracy'])} | {metric(c['evidence_f1_macro'])} | {metric(c['counterevidence_recall_macro'])} | {metric(c['both_recall'])} | {metric(c['insufficient_class_accuracy'])} | {metric(c['mean_total_tokens'],0)} | {metric(c['mean_elapsed_seconds'],2)} |

*Строго модельные ответы (`valid`/`valid_after_repair`), без `safe_fallback` и постобработанных ответов. Строгих ответов в базовом прогоне {n_model}/36; валидных после повторной попытки {n_repaired}, безопасных подстановок {n_fall}. Для стресса: {n_stress_model}/84 строгих ответов модели, {n_stress_fall} подстановок. Accuracy измеряет статус, а Evidence F1 — обоснованность; одно не заменяет другое. Пустые эталонные списки контрдоказательств учитываются при макроусреднении, поэтому Counter recall дополнительно следует смотреть по вопросам с непустым эталоном.

**Retrieval против полного контекста.** Покрытие эталонных утверждений BM25 top-15: {percent(rm.get('B_mean_gold_evidence_recall'))}. Режим B {'точнее' if (b['status_accuracy'] or 0)>(a['status_accuracy'] or 0) else 'не превосходит по точности'} A в этом запуске; сравнение токенов и времени приведено выше. По 12 вопросам нельзя уверенно оценить статистическую значимость разницы.

**p-adic и противоречия.** Accuracy режима B {comp}, чем C; подробное сопоставление по вопросам находится в `results/llm_metrics.json`. На вопросы `both` в эталонах приходится один случай, поэтому доля их распознавания чувствительна к единственной ошибке. p-adic меняет порядок и группировку, но не набор доступных доказательств; причинное объяснение выигрыша или проигрыша по этой выборке ограничено.

**Недостаточность данных.** Два вопроса требуют `insufficient`. Их точность по режимам указана в таблице. Доля уверенных ошибок (confidence ≥ 0,7) по таким вопросам: A {percent(a.get('confident_wrong_rate_on_insufficient'))}, B {percent(b.get('confident_wrong_rate_on_insufficient'))}, C {percent(c.get('confident_wrong_rate_on_insufficient'))}. Это отделяет обоснованный отказ от правдоподобной догадки.

## Стресс-тесты: только наблюдённые ответы LLM

**Перестановка документов (`random_state=42`).** Всего 36 запросов. В A меняется порядок исходных документов: изменение статуса относительно базового ответа на {percent(pe.get('model_only_answer_changed_rate'))} сопоставимых модельных пар, стабильность evidence ID {percent(pe.get('model_only_evidence_ids_stability'))}. В B/C порядок исходного массива документов не меняет получаемый retrieval-контекст, поэтому их 24 прогона являются **контролем повторяемости на одинаковом контексте**, а не эффектом перестановки документов. Частота смены статуса в этих контрольных парах: {percent(pe.get('same_context_control_model_only_status_changed_rate'))}. Это позволяет не путать позиционный эффект с вариативностью генерации.

**Удаление необходимой посылки.** Для Q03–Q06 подготовлено 12 запросов по трём режимам, удаляется один смысловой факт вместе с повторными формулировками. Доля совпадения с ожидаемым статусом: {percent(remove.get('model_only_matches_expected_status_rate'))} среди строгих ответов. Оставшиеся ошибки показывают, когда модель продолжает делать вывод без необходимого основания; наличие полного контекста и допустимых ID проверяется автоматически.

**Добавление противоречия.** Для 12 вопросов × 3 режима добавлено одно новое свидетельство или журнальная запись; тип источника и ожидание заданы заранее. Переход к `both` там, где он ожидался, достигнут в {percent(add.get('both_recall_on_expected_both'))} пригодных ответов; при слабом источнике правильный статус сохранён в {percent(add.get('unchanged_rate_when_source_is_weaker'))} случаев. Эти показатели описывают реакцию модели на вмешательство, а не точность разметки новых свидетельств.

## p-adic, confidence и ограничения

Вектор `[a0,a1,a2]` означает тип источника, логический статус и время. Тип источника важнее времени, поскольку машинный журнал имеет иной доказательный вес, чем неподтверждённое свидетельство; внутри ветви хронология помогает восстановить события. При перестановке уровней на «время первым» сначала объединялись бы события одной фазы, а надёжность стала бы вторичным признаком. Этот альтернативный порядок **не тестировался в обязательном прогоне**, поэтому улучшение ему не приписывается. P-adic может помогать группировать сведения одинаковой надёжности, но разносить спорящие свидетельства по разным ветвям.

Brier / ECE, вычисленные из confidence: A {metric(a.get('confidence_brier_score'))} / {metric(a.get('confidence_expected_calibration_error'))}; B {metric(b.get('confidence_brier_score'))} / {metric(b.get('confidence_expected_calibration_error'))}; C {metric(c.get('confidence_brier_score'))} / {metric(c.get('confidence_expected_calibration_error'))}. Малый объём данных и редкие классы не позволяют считать ECE надёжной оценкой калибровки. Несмотря на `temperature=0`, идентичные запросы могут давать различающиеся ответы на GPU. Ранее промпт обсуждался на этих же вопросах; результаты нельзя трактовать как независимую внешнюю проверку обобщающей способности. Итог — количественно разделены полнота поиска, качество доказательств, логический статус и устойчивость к изменениям контекста; превосходство p-adic не предполагалось заранее.
'''
    words=len(re.findall(r'\w+',report))
    if words > 1500: raise RuntimeError(f'Автоотчёт {words} слов, превышает лимит 1500')
    (ROOT/'report.md').write_text(report,encoding='utf-8')
    return words


def export_zip(name='lab6_submission_final.zip'):
    out = ROOT.parent / name
    exclude_names = {'.DS_Store', 'credentials.json', 'secrets.json'}
    exclude_parts = {'__pycache__','.git','.pytest_cache','.ipynb_checkpoints','.venv', 'venv'}
    with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=7) as z:
        for p in sorted(ROOT.rglob('*')):
            if not p.is_file() or p.name in exclude_names or p.suffix=='.pyc': continue
            rel=p.relative_to(ROOT)
            if set(rel.parts) & exclude_parts or p.name.startswith('.env'): continue
            z.write(p,str(Path('ai-detective')/rel))
    with zipfile.ZipFile(out) as z:
        bad=z.testzip()
        if bad: raise RuntimeError('ZIP повреждён: '+bad)
    return out


def main():
    d=validate()
    words=create_report(d)
    proc=subprocess.run([sys.executable,str(ROOT/'scripts/build_report_pdf.py')],cwd=ROOT, capture_output=True,text=True)
    if proc.returncode:
        raise RuntimeError('Не создан PDF: '+proc.stderr[-1500:])
    if not (ROOT/'report.pdf').exists():raise RuntimeError('Не найден report.pdf')
    summary={'status':'experiment_complete','model':MODEL,'base_observed':36,'stress_observed':84,
             'base_native_or_repaired':sum(r['validation_status'] in STRICT_MODEL for r in d['base']),
             'stress_native_or_repaired':sum(r['validation_status'] in STRICT_MODEL for r in d['stress']),
             'base_fallback':sum(r['validation_status']=='safe_fallback' for r in d['base']),
             'stress_fallback':sum(r['validation_status']=='safe_fallback' for r in d['stress']),
             'report_words':words,
             'manual_requirements':['confirm individual responsibilities in CONTRIBUTION.md','verify Streamlit UI in a browser after deployment','publish GitHub repository and check public access'],
             'team_gold_review':'confirmed_by_user_for_main_gold_and_stress_expected_statuses',
             'team_members':['Атанова Софья','Гольцман Глеб','Грищук Елизавета'],
             'submission_note':'Team gold review is recorded from user confirmation; individual role allocation, browser verification and remote Git publication remain external checks.'}
    (RESULTS/'final_validation.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    output=export_zip()
    print(json.dumps({**summary,'archive':str(output),'size_bytes':output.stat().st_size},ensure_ascii=False,indent=2))
    return output

if __name__=='__main__':main()
