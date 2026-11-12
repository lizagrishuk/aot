"""Бэкенд Ollama проверяется на локальном фиктивном HTTP-сервере, настоящая модель не нужна."""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import llm
from schemas import parse_model_answer

SEEN = {}
ANSWER = {"question_id": "Q01", "evidence_ids": ["C01"], "counterevidence_ids": [], "explanation": "C01 прямо указывает время.",
          "status": "entailed", "answer": "18:42:16", "confidence": 0.9}


class Handler(BaseHTTPRequestHandler):
    prompt_eval_count = 1234

    def log_message(self, *a): pass

    def _send(self, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        self._send({"models": [{"name": "qwen2.5:7b-instruct"}]})

    def do_POST(self):
        SEEN["payload"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self._send({"message": {"role": "assistant", "content": json.dumps(ANSWER, ensure_ascii=False)},
                    "done_reason": "stop", "prompt_eval_count": Handler.prompt_eval_count, "eval_count": 55})


@pytest.fixture()
def server(monkeypatch):
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("LLM_BACKEND", "ollama")
    monkeypatch.setenv("OLLAMA_URL", f"http://127.0.0.1:{srv.server_port}")
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    Handler.prompt_eval_count = 1234
    yield srv
    srv.shutdown()


def test_backend_selection_and_default_model(server):
    assert llm.active_backend() == "ollama"
    assert llm.selected_model_name() == "qwen2.5:7b-instruct"
    assert llm.backend_label() == "ollama_json_schema"


def test_request_contains_schema_options_and_prompt(server):
    raw = llm.generate_raw_answer("Q01", "Во сколько?", "C01 [D07, direct]: время 18:42:16", valid_claim_ids={"C01", "C02"})
    p = SEEN["payload"]
    assert p["stream"] is False and p["model"] == "qwen2.5:7b-instruct"
    assert p["options"]["temperature"] == 0.0 and p["options"]["seed"] == 42 and p["options"]["num_ctx"] == 32768
    assert raw["seed"] == 42 and raw["generation_mode"] == "temperature_zero_greedy"
    assert raw["request_nonce"] == p["messages"][0]["content"].split("\n", 1)[0].removeprefix("[request ").removesuffix("]")
    schema = p["format"]
    assert list(schema["properties"])[:5] == ["question_id", "evidence_ids", "counterevidence_ids", "explanation", "status"]
    assert schema["properties"]["evidence_ids"]["items"]["enum"] == ["C01", "C02"]
    assert p["messages"][0]["role"] == "system" and llm.SYSTEM_PROMPT in p["messages"][0]["content"]
    assert "Во сколько?" in p["messages"][1]["content"]
    assert raw["backend"] == "ollama_json_schema" and raw["api_key_used"] is False


def test_response_is_mapped_and_parses_with_strict_schema(server):
    raw = llm.generate_raw_answer("Q01", "q", "C01 [D07, direct]: x", valid_claim_ids={"C01"})
    assert raw["prompt_tokens"] == 1234 and raw["completion_tokens"] == 55 and raw["total_tokens"] == 1289
    assert raw["input_truncated"] is False and raw["finish_reason"] == "stop"
    assert parse_model_answer(raw["raw_text"], {"C01"}, "Q01").status.value == "entailed"


def test_request_nonce_is_stable_for_same_input_and_differs_when_input_changes(server):
    def prefix(question, context):
        llm.generate_raw_answer("Q01", question, context, valid_claim_ids={"C01"})
        return SEEN["payload"]["messages"][0]["content"]
    a1, a2 = prefix("q", "C01 x"), prefix("q", "C01 x")
    b = prefix("q", "C01 y")
    assert a1 == a2, "одинаковый запрос должен получать одинаковую cache-busting метку; это не гарантия одинакового ответа модели"
    assert a1.split("\n", 1)[0] != b.split("\n", 1)[0] and a1.split("\n", 1)[1] == b.split("\n", 1)[1]


def test_truncated_input_is_flagged(server):
    Handler.prompt_eval_count = 32768
    assert llm.generate_raw_answer("Q01", "q", "C01 x", valid_claim_ids={"C01"})["input_truncated"] is True


def test_warmup_checks_model_is_pulled(server, monkeypatch):
    assert llm.warmup_model()["model_loaded"] is True
    with pytest.raises(ValueError, match="использует только"):
        llm.warmup_model("no-such-model")


def test_unreachable_ollama_is_reported_without_fake_answer(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "ollama")
    monkeypatch.setenv("OLLAMA_URL", "http://127.0.0.1:1")
    with pytest.raises(RuntimeError, match="Ollama недоступна"):
        llm.generate_raw_answer("Q01", "q", "C01 x", valid_claim_ids={"C01"})
