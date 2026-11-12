from __future__ import annotations
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def load_documents():
    return load_jsonl(DATA / "documents.jsonl")

def load_claims():
    return load_jsonl(DATA / "claims.jsonl")

def load_questions():
    return load_jsonl(DATA / "questions.jsonl")

def dump_json(path: Path, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# Лимиты длины текстовых полей ответа. Схема допускает больше, чем просит промпт (200/250),
# чтобы ограничитель JSON не обрывал предложения на полуслове.
ANSWER_MAX_CHARS = 240
EXPLANATION_MAX_CHARS = 300
