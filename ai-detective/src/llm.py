"""One fixed local inference backend for Lab 6: Qwen2.5-7B-Instruct via Ollama.

All modes A/B/C and stress interventions use the exact same messages, model and
parameters. No hosted service or API token. No alternative model is downloaded.
"""
from __future__ import annotations
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from common import ANSWER_MAX_CHARS, EXPLANATION_MAX_CHARS

ROOT = Path(__file__).resolve().parents[1]
SYSTEM_PROMPT = (ROOT/'prompts/system_prompt.md').read_text(encoding='utf-8')
USER_TEMPLATE = (ROOT/'prompts/user_prompt_template.md').read_text(encoding='utf-8')
MODEL_NAME = 'qwen2.5:7b-instruct'
DEFAULT_SEED = 42
STATUS_VALUES = ['entailed','contradicted','both','insufficient']


def active_backend() -> str:
    return 'ollama'


def backend_label() -> str:
    return 'ollama_json_schema'


def selected_model_name(requested: str | None = None) -> str:
    # During notebook runs, the model is fixed; no accidental 14B/1.5B selection.
    name = requested or os.getenv('OLLAMA_MODEL', MODEL_NAME)
    if name != MODEL_NAME:
        raise ValueError(f'Лабораторная использует только {MODEL_NAME}; получено {name}')
    return MODEL_NAME


def build_messages(question_id: str, question: str, context: str,
                   repair_instruction: str | None = None) -> list[dict[str, str]]:
    repair = f'\nИСПРАВЬ ОШИБКУ ПРЕДЫДУЩЕГО ОТВЕТА: {repair_instruction}\n' if repair_instruction else ''
    user_text = USER_TEMPLATE.format(context=context,question_id=question_id,question=question,repair=repair)
    return [{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':user_text}]


def _allowed_ids_from_context(context: str, valid_claim_ids: set[str] | list[str] | None) -> list[str]:
    if valid_claim_ids is not None:
        ids = {str(x) for x in valid_claim_ids}
    else:
        ids = set(re.findall(r'\bC\d{2}\b',context)) | set(re.findall(r'\bCX\d+\b',context))
    if not ids: raise ValueError('No claim IDs were supplied or found in the context')
    return sorted(ids)


def response_schema(question_id: str, allowed_ids: list[str]) -> dict:
    return {
        'type':'object',
        'properties': {
            'question_id':{'type':'string','enum':[question_id]},
            'evidence_ids':{'type':'array','items':{'type':'string','enum':allowed_ids},'maxItems':4},
            'counterevidence_ids':{'type':'array','items':{'type':'string','enum':allowed_ids},'maxItems':4},
            'explanation':{'type':'string','maxLength':EXPLANATION_MAX_CHARS},
            'status':{'type':'string','enum':STATUS_VALUES},
            'answer':{'type':'string','maxLength':ANSWER_MAX_CHARS},
            'confidence':{'type':'number','minimum':0,'maximum':1},
        },
        'required':['question_id','evidence_ids','counterevidence_ids','explanation','status','answer','confidence'],
        'additionalProperties':False,
    }


def _ollama_url() -> str:
    return os.getenv('OLLAMA_URL','http://127.0.0.1:11434').rstrip('/')


def _ollama_request(path: str, payload: dict | None, timeout: float) -> dict:
    data=None if payload is None else json.dumps(payload,ensure_ascii=False).encode('utf-8')
    request=urllib.request.Request(_ollama_url()+path,data=data,
                                    headers={'Content-Type':'application/json'},
                                    method='GET' if data is None else 'POST')
    try:
        with urllib.request.urlopen(request,timeout=timeout) as response:
            return json.loads(response.read().decode('utf-8'))
    except (urllib.error.URLError,TimeoutError,OSError,json.JSONDecodeError) as exc:
        raise RuntimeError(f'Ollama недоступна по адресу {_ollama_url()}: {exc}. Запущена ли Ollama?') from exc


def warmup_model(model: str | None = None) -> dict[str,Any]:
    name=selected_model_name(model)
    data=_ollama_request('/api/tags',None,15)
    have={m.get('name','') for m in data.get('models',[])}
    if name not in have and f'{name}:latest' not in have:
        raise RuntimeError(f'Модель {name} не скачана: ollama pull {name}')
    return {'model':name,'model_loaded':True,'backend':'ollama',
            'api_key_required':False,'json_constrained_decoding':True}


def request_nonce(question_id: str, question: str, context: str,
                  repair_instruction: str | None = None) -> str:
    text='\n'.join([question_id,question,context,repair_instruction or ''])
    return hashlib.sha1(text.encode('utf-8')).hexdigest()[:8]


def generate_raw_answer(question_id: str, question: str, context: str,
                        api_key: str | None = None, model: str | None = None,
                        temperature: float = 0.0, valid_claim_ids: set[str] | list[str] | None = None,
                        repair_instruction: str | None = None) -> dict[str,Any]:
    if float(temperature) != 0.0:
        raise ValueError('Эксперимент фиксирует temperature=0 во всех режимах')
    name=selected_model_name(model)
    allowed_ids=_allowed_ids_from_context(context,valid_claim_ids)
    msgs=build_messages(question_id,question,context,repair_instruction)
    nonce=request_nonce(question_id,question,context,repair_instruction)
    # Stable nonce prevents silent prompt-cache reuse from corrupting token comparisons.
    msgs[0]={'role':'system','content':f'[request {nonce}]\n'+msgs[0]['content']}
    num_ctx=int(os.getenv('OLLAMA_NUM_CTX','32768'))
    request={
        'model':name,'messages':msgs,'stream':False,
        'format':response_schema(question_id,allowed_ids),
        'options':{'temperature':0.0,'seed':DEFAULT_SEED,'num_ctx':num_ctx,'num_predict':512},
    }
    start=time.perf_counter()
    data=_ollama_request('/api/chat',request,float(os.getenv('OLLAMA_TIMEOUT','900')))
    elapsed=time.perf_counter()-start
    raw_text=(data.get('message') or {}).get('content','').strip()
    prompt_tokens=int(data.get('prompt_eval_count') or 0)
    completion_tokens=int(data.get('eval_count') or 0)
    truncated=prompt_tokens >= num_ctx - 8
    return {'raw_text':raw_text,'model':name,'backend':'ollama_json_schema',
            'temperature':0.0,'seed':DEFAULT_SEED,'request_nonce':nonce,
            'generation_mode':'temperature_zero_greedy',
            'elapsed_seconds':round(elapsed,4),
            'prompt_tokens':prompt_tokens,'original_prompt_tokens':prompt_tokens,
            'completion_tokens':completion_tokens,'total_tokens':prompt_tokens+completion_tokens,
            'finish_reason':data.get('done_reason') or 'stop',
            'api_key_used':False,'input_token_limit':num_ctx,
            'input_truncated':truncated,'attention_backend':'ollama',
            'memory_fallback_count':0,'allowed_claim_ids':allowed_ids,'json_constrained':True}
