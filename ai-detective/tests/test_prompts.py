"""Промпты: один источник правды, нет подсказок под конкретное досье, схема согласована с форматом."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from llm import SYSTEM_PROMPT, USER_TEMPLATE, build_messages, response_schema, STATUS_VALUES
from common import ANSWER_MAX_CHARS, EXPLANATION_MAX_CHARS, load_documents, load_claims

# слова и идентификаторы, которые есть в досье и не должны попадать в промпт (утечка ответа)
LEAK_TERMS = ["message_id", "unique_message_count", "usr_4821", "INC-260314", "MOD-4.2", "19:10", "18:42", "msg_771", "Соколов", "Контур"]


def test_prompts_do_not_leak_dossier_specifics():
    text = SYSTEM_PROMPT + USER_TEMPLATE
    assert [t for t in LEAK_TERMS if t in text] == []


def test_prompt_does_not_reuse_dossier_vocabulary_in_examples():
    docs_text = " ".join(d["text"] for d in load_documents())
    for name in ("Ким", "Ветер", "Лена", "Нора", "Олег", "Гора", "Река"):
        assert name not in docs_text, "примеры промпта должны быть из другой предметной области"


def test_user_template_has_exactly_the_placeholders_the_code_fills():
    assert sorted(set(re.findall(r"\{(\w+)\}", USER_TEMPLATE))) == ["context", "question", "question_id", "repair"]


def test_template_does_not_ask_for_fields_missing_in_schema():
    assert "reasoning_summary" not in USER_TEMPLATE + SYSTEM_PROMPT


def test_system_prompt_defines_all_statuses_and_has_one_example_per_status():
    for s in STATUS_VALUES:
        assert f"`{s}`" in SYSTEM_PROMPT
        assert f'"status": "{s}"' in SYSTEM_PROMPT, f"нет примера для {s}"


def test_prompt_json_key_order_matches_schema_order():
    schema = response_schema("Q01", ["C01", "C02"])
    order = list(schema["properties"])
    assert order == ["question_id", "evidence_ids", "counterevidence_ids", "explanation", "status", "answer", "confidence"]
    example = json.loads(re.search(r'\{"question_id": "Q0".*\}', SYSTEM_PROMPT).group(0))
    assert list(example) == order
    assert set(schema["required"]) == set(order)


def test_schema_limits_leave_room_for_whole_sentences():
    props = response_schema("Q01", ["C01"])["properties"]
    assert props["explanation"]["maxLength"] == EXPLANATION_MAX_CHARS >= 250
    assert props["answer"]["maxLength"] == ANSWER_MAX_CHARS >= 200


def test_question_comes_after_context():
    msgs = build_messages("Q01", "ВОПРОС-МАРКЕР?", "КОНТЕКСТ-МАРКЕР", None)
    user = msgs[1]["content"]
    assert user.index("КОНТЕКСТ-МАРКЕР") < user.index("ВОПРОС-МАРКЕР?")
    assert msgs[0]["role"] == "system" and msgs[0]["content"] == SYSTEM_PROMPT


def test_repair_instruction_is_included_only_when_given():
    assert "ИСПРАВЬ" not in build_messages("Q01", "q", "c", None)[1]["content"]
    assert "ИСПРАВЬ ОШИБКУ ПРЕДЫДУЩЕГО ОТВЕТА: x" in build_messages("Q01", "q", "c", "x")[1]["content"]


def test_context_with_braces_is_not_interpreted_as_template():
    assert "{boom}" in build_messages("Q01", "q", "контекст {boom} json {\"a\": 1}", None)[1]["content"]


def test_claim_ids_in_schema_are_enum_restricted():
    props = response_schema("Q07", ["C01", "C02"])["properties"]
    assert props["question_id"]["enum"] == ["Q07"]
    assert props["evidence_ids"]["items"]["enum"] == ["C01", "C02"]


def test_iteration2_rules_present():
    for rule in ["Вопросы да/нет", "→ `contradicted` (не `entailed`)", "а не `both`", "заявления без независимой проверки"]:
        assert rule in SYSTEM_PROMPT, rule


def test_prompt_has_examples_for_yes_no_and_log_vs_testimony():
    assert SYSTEM_PROMPT.count('"status": "contradicted"') >= 2  # «Верно ли» и «Мог ли»
    assert 'Записи журнала надёжнее пересказа' in SYSTEM_PROMPT
