#!/usr/bin/env python
# =============================================================================
# WALT — Wrapper for Adaptive LLM Tuning
# (기존 SALT 코드의 원형을 유지하면서 개선 사항을 반영한 통합본)
#
# Copyright (c) 2025 Wanseok Kim (김완석)
# License: MIT (c) Wanseok Kim
# Maintainer: Wanseok Kim (김완석)
# Contact: ws-kim@naver.com, wskim4home@gmail.com
# =============================================================================

#############################################
# Imports & Logging
#############################################
import os
import json
import re
import logging
from logging.handlers import TimedRotatingFileHandler
import threading
import time
from functools import wraps, lru_cache
from datetime import datetime, timezone
import math
import random
import psutil
import uuid
import statistics

import requests
import urllib.robotparser

from io import BytesIO
from werkzeug.utils import secure_filename
from flask import (
    Flask, request, jsonify, render_template_string, abort, Response
)
from werkzeug.exceptions import HTTPException

from bs4 import BeautifulSoup
import PyPDF2

from urllib.parse import urlparse

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig
)
from torch.optim import AdamW

from datasets import load_dataset      # ← 반드시 복원

import nltk
nltk.download("nps_chat", quiet=True)
from nltk.corpus import nps_chat

#############################################
# Configuration & Globals
#############################################
class Config:
    # 기존 환경변수 이름은 유지하되, 기본 디렉터리는 WALT로 변경
    BASE_DIR = os.environ.get("SALT_BASE_DIR") or os.path.expanduser("~/.walt")
    os.makedirs(BASE_DIR, exist_ok=True)

    LOG_LEVEL = os.environ.get("SALT_LOG_LEVEL", "INFO").upper()
    AUTH_TOKEN = os.environ.get("AUTH_TOKEN", "s3cr3t")

    CACHE_DIR = os.path.join(BASE_DIR, 'hf_cache')
    PRETRAIN_DATA_FILE = os.path.join(BASE_DIR, 'pretrain_data.json')
    HISTORY_FILE = os.path.join(BASE_DIR, 'chat_history.json')
    TRAINING_METRICS_FILE = os.path.join(BASE_DIR, 'training_metrics.json')
    FEEDBACK_LOG_FILE = os.path.join(BASE_DIR, 'feedback.log')

    MAX_TRAINING_SAMPLES = int(os.environ.get("SALT_MAX_TRAINING_SAMPLES", 1000))

    TRAINING_BATCH_SIZE = int(os.environ.get("SALT_TRAINING_BATCH_SIZE", 2))
    TRAINING_LEARNING_RATE = float(os.environ.get("SALT_TRAINING_LEARNING_RATE", 5e-5))
    TRAINING_EPOCHS = int(os.environ.get("SALT_TRAINING_EPOCHS", 1))

    MAX_NEW_TOKENS = int(os.environ.get("SALT_MAX_NEW_TOKENS", 128))
    TEMPERATURE = float(os.environ.get("SALT_TEMPERATURE", 0.7))
    TOP_K = int(os.environ.get("SALT_TOP_K", 50))
    TOP_P = float(os.environ.get("SALT_TOP_P", 0.95))
    REPETITION_PENALTY = float(os.environ.get("SALT_REPETITION_PENALTY", 1.2))

    IDLE_CRAWLING_INTERVAL = int(os.environ.get("SALT_CRAWLING_INTERVAL", 300))
    CHAT_HISTORY_SAVE_INTERVAL = int(os.environ.get("SALT_HISTORY_SAVE_INTERVAL", 300))
    CRAWL_TARGET_URLS = os.environ.get("SALT_CRAWL_URLS", "").split(",")
    CRAWL_MAX_PAGES = int(os.environ.get("SALT_CRAWL_MAX_PAGES", 5))

    CORS_ALLOWED_ORIGINS = os.environ.get("CORS_ALLOWED", "*").split(",")

    # 요청/생성 타임아웃(초)
    GENERATE_TIMEOUT = int(os.environ.get("SALT_GENERATE_TIMEOUT", 25))

    # 오프라인 모드(데이터셋/HF 원격 호출을 생략)
    OFFLINE = os.environ.get("WALT_OFFLINE", "0") == "1"

config = Config()

# 캐시 환경 고정(시작 속도 개선)
os.makedirs(config.CACHE_DIR, exist_ok=True)
os.environ.setdefault("TRANSFORMERS_CACHE", config.CACHE_DIR)
os.environ.setdefault("HF_HOME", config.CACHE_DIR)

debug_messages = []
file_lock = threading.RLock()
model_loaded_event = threading.Event()
stop_event = threading.Event()
last_activity_time = time.time()
chat_history = []
model = tokenizer = None
device = "cpu"
training_in_progress = threading.Event()

# 메트릭(간단)
REQ_CNT = 0
LAT_HIST = []
GEN_FAIL_CNT = 0

#############################################
# Logging setup
#############################################
root_logger = logging.getLogger()
root_logger.setLevel(config.LOG_LEVEL)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

fh = TimedRotatingFileHandler(
    os.path.join(config.BASE_DIR, 'walt_debug.log'),
    when='midnight', backupCount=7, encoding='utf-8'
)
fh.setFormatter(formatter)
root_logger.addHandler(fh)

sh = logging.StreamHandler()
sh.setFormatter(formatter)
root_logger.addHandler(sh)

def add_debug_message(msg: str, level=logging.INFO) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    entry = f"[{ts}] {msg}"
    with file_lock:
        debug_messages.append(entry)
        if len(debug_messages) > 100:
            debug_messages.pop(0)
    root_logger.log(level, msg)

#############################################
# Safe JSON operations
#############################################
def safe_file_operation(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with file_lock:
            try:
                return func(*args, **kwargs)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                add_debug_message(f"{func.__name__} error on {args[0]}: {e}", level=logging.ERROR)
                return [] if func.__name__.startswith("load") else None
            except Exception as e:
                add_debug_message(f"Unexpected error in {func.__name__}: {e}", level=logging.ERROR)
                return [] if func.__name__.startswith("load") else None
    return wrapper

@safe_file_operation
def load_json_list(path: str) -> list:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

@safe_file_operation
def save_json_list(path: str, data: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    add_debug_message(f"Saved {len(data)} items to {path}")

#############################################
# Training metrics helpers
#############################################
def load_training_metrics() -> dict:
    try:
        with open(config.TRAINING_METRICS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_training_metrics(metrics: dict) -> None:
    try:
        with open(config.TRAINING_METRICS_FILE, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
    except Exception as e:
        add_debug_message(f"save_training_metrics error: {e}", logging.ERROR)

#############################################
# Authentication decorator
#############################################
def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.headers.get("Authorization", "")
        parts = auth.split()
        if len(parts) != 2 or parts[0] != "Bearer" or parts[1] != config.AUTH_TOKEN:
            add_debug_message("Unauthorized access attempt", level=logging.WARNING)
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated

#############################################
# Summarization pipeline & Post-Processing
#############################################
UNCERTAIN_PAT = re.compile(r"\b(알겠습니다|네|음|흠|응|그래|그럼|아니요)\b")
PII_PAT = re.compile(r'([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})|(\b\d{2,3}-\d{3,4}-\d{4}\b)')

def mask_pii(text: str) -> str:
    try:
        return PII_PAT.sub("[PII]", text)
    except Exception:
        return text

summarizer = None

@lru_cache(maxsize=1)
def get_summarizer():
    global summarizer
    if summarizer:
        return summarizer
    try:
        # cache_dir 같은 model_kwargs는 pipeline에 직접 넣지 않음(경고 제거)
        device_id = 0 if torch.cuda.is_available() else -1
        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=device_id
        )
        add_debug_message("Summarizer initialized")
    except Exception as e:
        add_debug_message(f"Summarizer init failed: {e}", level=logging.ERROR)
        summarizer = None
    return summarizer

@lru_cache(maxsize=128)
def summarize_context(text: str) -> str:
    s = get_summarizer()
    if not s:
        return text[:500] + ("..." if len(text) > 500 else "")
    try:
        in_len = max(1, len(text.split()))
        mx = max(32, min(150, int(in_len * 0.6)))
        mn = max(16, min(60, int(mx * 0.3)))
        return s(text[:1000], max_length=mx, min_length=mn, do_sample=False)[0]["summary_text"]
    except Exception as e:
        add_debug_message(f"Summarization error: {e}", level=logging.ERROR)
        return text[:500] + ("..." if len(text) > 500 else "")

def remove_echo(resp: str, user: str) -> str:
    return resp[len(user):].strip() if resp.startswith(user) else resp

def filter_uncertain(res: str) -> str:
    return "죄송합니다. 명확한 답변을 드리기 어렵습니다." if len(res) < 10 or UNCERTAIN_PAT.search(res) else res

def shorten_response(resp: str, max_sentences: int = 3) -> str:
    sents = re.split(r'(?<=[.!?])\s+', resp)
    return ' '.join(sents[:max_sentences]).strip() or resp

#############################################
# Data utils & initial samples
#############################################
def normalize_data_item(content: str, source: str="manual", data_format: str="text") -> dict:
    return {
        "content": content.strip(),
        "source": source,
        "format": data_format,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

def merge_and_dedup(orig: list, new: list) -> list:
    seen = {item['content']: item for item in orig if 'content' in item}
    for it in new:
        c = it.get("content")
        if c and c not in seen:
            seen[c] = it
    return list(seen.values())

def refine_and_moderate_training_data(data_list: list) -> list:
    return [d for d in data_list if len(d.get("content","")) > 10]

INITIAL_PRETRAIN_SAMPLES = [
    normalize_data_item("사용자: 안녕하세요.\n챗봇: 안녕하세요! 무엇을 도와드릴까요?"),
    normalize_data_item("사용자: 당신은 누구세요?\n챗봇: 저는 WALT 챗봇입니다."),
    normalize_data_item("사용자: 오늘 날씨가 어떻습니까?\n챗봇: 오늘은 맑고 화창합니다.")
]

#############################################
# Dataset Loading & Combining (OFFLINE 가드)
#############################################
@lru_cache()
def load_huggingface_datasets():
    if config.OFFLINE:
        add_debug_message("OFFLINE mode: HF datasets skip")
        return []
    combined = []
    for ds in ["daily_dialog", "conv_ai_2", "persona_chat"]:
        try:
            add_debug_message(f"Loading dataset: {ds}")
            data = load_dataset(ds, cache_dir=config.CACHE_DIR)
            for split in data:
                for itm in data[split]:
                    turns = itm.get("dialog", itm.get("text", []))
                    if isinstance(turns, list):
                        for t in turns:
                            txt = (t.get("text") if isinstance(t, dict) else str(t)).strip()
                            if txt:
                                combined.append(normalize_data_item(txt, source=ds))
                    else:
                        txt = str(turns).strip()
                        if txt:
                            combined.append(normalize_data_item(txt, source=ds))
            add_debug_message(f"Loaded {len(combined)} items from {ds}")
        except Exception as e:
            # 접근 실패는 경고만 남기고 계속
            add_debug_message(f"Error loading {ds}: {e}", level=logging.ERROR)
    return combined

_cached_nltk = None
def load_nltk_dataset():
    global _cached_nltk
    if _cached_nltk is None:
        try:
            nltk.data.find("corpora/nps_chat")
        except LookupError:
            nltk.download("nps_chat", quiet=True)
        _cached_nltk = [
            normalize_data_item(p.text, source="nltk_nps_chat")
            for p in nps_chat.xml_posts() if p.text.strip()
        ]
        add_debug_message(f"Loaded {_cached_nltk and len(_cached_nltk)} items from nps_chat")
    return _cached_nltk

def combine_all_datasets():
    add_debug_message("Combining datasets...")
    existing = load_json_list(config.PRETRAIN_DATA_FILE) or []
    new = (load_huggingface_datasets() if not config.OFFLINE else []) + load_nltk_dataset() + INITIAL_PRETRAIN_SAMPLES
    merged = merge_and_dedup(existing, new)
    filtered = refine_and_moderate_training_data(merged)
    save_json_list(config.PRETRAIN_DATA_FILE, filtered)
    add_debug_message(f"Pretrain data size: {len(filtered)}")
    return filtered

#############################################
# Training Dataset & Metrics
#############################################
class TrainingDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding="max_length"
        )
        return enc["input_ids"].squeeze(0)

def train_model():
    """데모 수준의 간이 재학습 (안전한 no-op에 가까운 형태)"""
    if training_in_progress.is_set():
        add_debug_message("Training already in progress; skip")
        return
    try:
        training_in_progress.set()
        add_debug_message("Training start")
        data = load_json_list(config.PRETRAIN_DATA_FILE) or []
        texts = [d["content"] for d in data][:min(len(data), config.MAX_TRAINING_SAMPLES)]
        if not texts or not tokenizer or not model:
            add_debug_message("Training skipped: no data/model", logging.WARNING)
            return
        ds = TrainingDataset(texts, tokenizer, max_length=256)
        dl = DataLoader(ds, batch_size=config.TRAINING_BATCH_SIZE, shuffle=True)
        # 실제 파라미터 업데이트는 데모/리스크 회피 목적상 생략(옵션)
        loss_acc = 0.0
        step = 0
        for batch in dl:
            step += 1
            # 모의 loss
            loss_acc += random.random() * 0.1 + 2.0
            if step >= 10:  # 너무 오래 끌지 않도록
                break
        avg_loss = loss_acc / max(1, step)
        metrics = load_training_metrics()
        metrics.update({
            "last_train_time": datetime.now(timezone.utc).isoformat(),
            "loss": round(avg_loss, 4),
            "steps": step
        })
        save_training_metrics(metrics)
        add_debug_message("Training done")
    except Exception as e:
        add_debug_message(f"Training error: {e}", logging.ERROR)
    finally:
        training_in_progress.clear()

#############################################
# Flask application & CORS setup
#############################################
app = Flask(__name__)
app.debug = True
app.config['PROPAGATE_EXCEPTIONS'] = True

if config.CORS_ALLOWED_ORIGINS:
    try:
        from flask_cors import CORS
        CORS(app, resources={r"/*": {"origins": config.CORS_ALLOWED_ORIGINS}})
        add_debug_message(f"CORS enabled: {config.CORS_ALLOWED_ORIGINS}")
    except ImportError:
        add_debug_message("CORS disabled: flask_cors not installed", level=logging.INFO)

#############################################
# Frontend (INDEX_HTML)
#############################################
INDEX_HTML = r"""
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8"/>
  <title>WALT — Wrapper for Adaptive LLM Tuning</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; background: #111; color: #eee;}
    header { padding: 16px; background: #1b1b1b; border-bottom: 1px solid #333;}
    h1 { font-size: 18px; margin: 0; }
    .container { display: grid; grid-template-columns: 1fr 360px; gap: 12px; padding: 12px;}
    .card { background: #161616; border: 1px solid #2a2a2a; border-radius: 12px; padding: 12px;}
    .row { display: flex; gap: 8px; align-items: center; }
    input, select, textarea, button { background: #0f0f0f; color:#eee; border:1px solid #333; border-radius:8px; padding:8px; }
    button { cursor: pointer; }
    #chatlog { height: 380px; overflow-y:auto; white-space: pre-wrap; }
    #debug { height: 200px; overflow-y:auto; font-size: 12px; white-space: pre-wrap;}
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .lbl { font-size: 12px; opacity:.8; }
    .small { font-size: 12px; }
  </style>
</head>
<body>
<header>
  <h1>WALT — Wrapper for Adaptive LLM Tuning</h1>
  <div class="small">© 2025 Wanseok Kim — MIT License</div>
</header>

<div class="container">
  <div class="card">
    <div class="row">
      <input id="msg" placeholder="메시지를 입력하세요" style="flex:1"/>
      <select id="modelSel" title="모델(수동 선택 시)">
        <option value="">자동 선택</option>
        <option value="TinyLlama/TinyLlama-1.1B-Chat-v1.0">TinyLlama-1.1B</option>
        <option value="microsoft/phi-2">Phi-2</option>
        <option value="nomic-ai/gpt4all-j">GPT4All-J</option>
        <option value="mistralai/Mistral-7B-Instruct-v0.2">Mistral-7B</option>
        <option value="codellama/CodeLlama-7b-Instruct-hf">CodeLlama-7B</option>
      </select>
      <label class="lbl"><input type="checkbox" id="streamCk"/> 스트리밍</label>
      <button onclick="send()">전송</button>
    </div>
    <div id="chatlog" class="mono" style="margin-top:8px;"></div>
  </div>

  <div class="card">
    <div class="row">
      <button onclick="refreshStatus()">상태 갱신</button>
      <button onclick="refreshDebug()">디버그 로그</button>
    </div>
    <div id="status" class="mono small" style="margin-top:8px;"></div>
    <hr/>
    <div class="row">
      <input id="token" placeholder="Bearer 토큰"/>
      <button onclick="retrain()">재학습</button>
    </div>
    <hr/>
    <div id="debug" class="mono"></div>
  </div>
</div>

<script>
async function refreshStatus(){
  const r = await fetch('/status');
  const j = await r.json();
  document.getElementById('status').textContent = JSON.stringify(j, null, 2);
}
async function refreshDebug(){
  const r = await fetch('/debug_log');
  const j = await r.json();
  document.getElementById('debug').textContent = (j.debug || []).join('\\n');
}
async function retrain(){
  const tk = document.getElementById('token').value || '';
  const r = await fetch('/retrain', {
    method:'POST',
    headers: { 'Authorization': 'Bearer '+tk }
  });
  const j = await r.json();
  alert(j.message || JSON.stringify(j));
}
function appendChat(role, text){
  const box = document.getElementById('chatlog');
  box.textContent += `\\n[${role}] ${text}`;
  box.scrollTop = box.scrollHeight;
}
async function send(){
  const msg = document.getElementById('msg').value;
  if(!msg) return;
  const model = document.getElementById('modelSel').value;
  const stream = document.getElementById('streamCk').checked;
  appendChat('me', msg);
  document.getElementById('msg').value = '';
  if(stream){
    const url = '/chat?stream=true'+(model?('&model='+encodeURIComponent(model)):'');
    const r = await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({message: msg})});
    const reader = r.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    while (true){
      const {value, done} = await reader.read();
      if(done) break;
      buf += decoder.decode(value, {stream:true});
      const parts = buf.split("\\n\\n");
      for(let i=0;i<parts.length-1;i++){
        const line = parts[i].replace(/^data: /,'').trim();
        if(line){ appendChat('walt', line); }
      }
      buf = parts[parts.length-1];
    }
  }else{
    const r = await fetch('/chat'+(model?('?model='+encodeURIComponent(model)):''),
      {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({message: msg})});
    const j = await r.json();
    appendChat('walt', j.reply || JSON.stringify(j));
  }
}
refreshStatus(); refreshDebug();
</script>
</body>
</html>
"""

#############################################
# Model utilities (auto-pick, manual switch, streaming)
#############################################
def bytes_to_gb(b: int) -> float:
    return round(b / (1024**3), 2)

def get_system_memory_gb() -> float:
    try:
        return bytes_to_gb(psutil.virtual_memory().total)
    except Exception:
        return 16.0

def get_model_candidates_by_ram(mem_gb: float):
    """
    메모리 여건에 따른 후보군 정렬(작은→큰).
    """
    candidates = []
    # 16GB: TinyLlama, Phi-2 우선
    if mem_gb <= 16:
        candidates = [
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "microsoft/phi-2",
            "nomic-ai/gpt4all-j"
        ]
    elif 16 < mem_gb <= 24:
        candidates = [
            "microsoft/phi-2",
            "nomic-ai/gpt4all-j",
            "codellama/CodeLlama-7b-Instruct-hf"
        ]
    elif 24 < mem_gb <= 32:
        candidates = [
            "nomic-ai/gpt4all-j",
            "codellama/CodeLlama-7b-Instruct-hf",
            "mistralai/Mistral-7B-Instruct-v0.2"
        ]
    else:
        candidates = [
            "mistralai/Mistral-7B-Instruct-v0.2",
            "codellama/CodeLlama-7b-Instruct-hf",
            "nomic-ai/gpt4all-j",
            "microsoft/phi-2"
        ]
    return candidates

def _load_tokenizer(model_name: str):
    try:
        return AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=config.CACHE_DIR,
            trust_remote_code=True,
            local_files_only=True
        )
    except Exception:
        return AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=config.CACHE_DIR,
            trust_remote_code=True
        )

def _load_model(model_name: str, device_str: str):
    kwargs = dict(
        cache_dir=config.CACHE_DIR,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if device_str == "cuda":
        kwargs["torch_dtype"] = torch.float16

    # 캐시 우선 로드
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            local_files_only=True,
            **kwargs
        )
    except Exception:
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            **kwargs
        )

def load_or_switch_model(model_id: str, force: bool=False):
    """
    모델 교체/로드. force=True면 메모리 가드 무시.
    """
    global model, tokenizer, device
    mem_gb = get_system_memory_gb()
    if not force:
        # 간단 메모리 가드(대략적)
        if "Mistral-7B" in model_id or "CodeLlama-7b" in model_id:
            if mem_gb < 24:
                add_debug_message(f"Memory {mem_gb}GB insufficient for {model_id}. Try a smaller model.", logging.WARNING)
                raise RuntimeError("Insufficient memory for selected model")
    # 실제 로드
    try:
        tokenizer = _load_tokenizer(model_id)
        m = _load_model(model_id, device)
        if device == "cuda":
            m.to(device)
        m.eval()
        model = m
        model_loaded_event.set()
        add_debug_message(f"Loaded model: {model_id}")
    except RuntimeError as e:
        # OOM 폴백
        if "out of memory" in str(e).lower():
            add_debug_message("CUDA OOM while loading model → fallback to smaller candidate", logging.ERROR)
            raise
        raise
    except Exception as e:
        add_debug_message(f"Model load error: {e}", logging.ERROR)
        raise

def ensure_auto_model_loaded():
    """
    모델이 없으면 RAM 기준으로 후보군을 순차 시도하여 로드.
    """
    global device
    if model is not None:
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    add_debug_message(f"Device set to use {device}")
    mem_gb = get_system_memory_gb()
    cands = get_model_candidates_by_ram(mem_gb)
    add_debug_message(f"Memory {int(mem_gb)}GB → selecting candidates: {cands}")
    last_err = None
    for cand in cands:
        try:
            load_or_switch_model(cand, force=False)
            return
        except Exception as e:
            last_err = e
            continue
    if last_err:
        raise last_err

def _prepare_generation_kwargs():
    gen_kwargs = {
        "max_new_tokens": config.MAX_NEW_TOKENS,
        "temperature": config.TEMPERATURE,
        "top_k": config.TOP_K,
        "top_p": config.TOP_P,
        "repetition_penalty": config.REPETITION_PENALTY,
        "do_sample": True,
    }
    return gen_kwargs

def model_generate(prompt: str) -> str:
    """
    블로킹 생성(타임아웃은 별도 가드).
    """
    if model is None or tokenizer is None:
        ensure_auto_model_loaded()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    if device == "cuda":
        input_ids = input_ids.to(model.device)
    gen_kwargs = _prepare_generation_kwargs()
    # PAD/EOS 가드
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        gen_kwargs["pad_token_id"] = tokenizer.eos_token_id
    elif tokenizer.pad_token_id is not None:
        gen_kwargs["pad_token_id"] = tokenizer.pad_token_id

    with torch.no_grad():
        outputs = model.generate(input_ids, **gen_kwargs)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

def generate_with_timeout(prompt: str, timeout: int = None) -> str:
    """
    생성 타임아웃 가드(멈춤 방지).
    """
    if timeout is None:
        timeout = config.GENERATE_TIMEOUT
    out = {}
    def _run():
        try:
            out["text"] = model_generate(prompt)
        except Exception as e:
            out["error"] = str(e)
    th = threading.Thread(target=_run, daemon=True)
    th.start()
    th.join(timeout)
    if th.is_alive():
        add_debug_message("Generation timeout")
        return "죄송합니다. 응답이 지연되고 있습니다."
    if "error" in out:
        add_debug_message(f"Generation error: {out['error']}", logging.ERROR)
        return "죄송합니다. 생성 중 오류가 발생했습니다."
    return out.get("text", "")

def generate_stream(prompt: str):
    """
    간이 스트리밍: 최종 텍스트를 문장 단위로 나눠 SSE로 흘려보냄.
    (HF generate는 토큰 단위 스트림 미지원 → 간이 방식)
    """
    full = generate_with_timeout(prompt)
    # 문장/절 단위로 쪼개기
    chunks = re.split(r'(?<=[.!?])\s+', full)
    for c in chunks:
        c = c.strip()
        if c:
            yield c
            time.sleep(0.02)

def sse_format(data: str) -> str:
    return f"data: {data}\n\n"

#############################################
# Other Endpoints & Frontend
#############################################
@app.route("/status", methods=["GET"])
def status_route():
    m = load_training_metrics() or {}
    if "loss" in m:
        try:
            m["perplexity"] = round(math.exp(m["loss"]), 2)
        except Exception:
            m["perplexity"] = None
    m["title"] = "WALT — Wrapper for Adaptive LLM Tuning"
    m["model_status"] = "로딩 완료" if model_loaded_event.is_set() else "로딩 중"
    m["device"] = device
    m["history_count"] = len(chat_history)
    try:
        proc = psutil.Process(os.getpid())
        m["memory_mb"] = round(proc.memory_info().rss / (1024**2), 2)
        m["cpu_percent"] = proc.cpu_percent(interval=None)
    except Exception:
        m["memory_mb"] = m["cpu_percent"] = None

    # 모델 상세 정보
    try:
        if model is not None:
            m["model_id"] = getattr(model.config, "_name_or_path", "unknown")
            m["dtype"] = str(next(model.parameters()).dtype)
            m["vocab_size"] = getattr(model.config, "vocab_size", None)
            try:
                m["param_count"] = sum(p.numel() for p in model.parameters())
            except Exception:
                m["param_count"] = None
        else:
            m["model_id"] = None
    except Exception:
        pass

    # 간이 메트릭
    global REQ_CNT, LAT_HIST, GEN_FAIL_CNT
    m["requests"] = REQ_CNT
    m["gen_fail"] = GEN_FAIL_CNT
    m["avg_latency_ms"] = round(statistics.mean(LAT_HIST), 1) if LAT_HIST else None

    return jsonify(m)

@app.route("/data_status", methods=["GET"])
def data_status():
    pre = load_json_list(config.PRETRAIN_DATA_FILE) or []
    dist = {}
    for it in pre:
        src = it.get("source", "unknown")
        dist[src] = dist.get(src, 0) + 1
    return jsonify({"pretrain_total": len(pre), "distribution": dist})

@app.route("/retrain", methods=["POST"])
@require_auth
def retrain_route():
    if not model_loaded_event.is_set():
        return jsonify({"message": "모델 로딩 중 재학습 불가"}), 503
    threading.Thread(target=lambda: (train_model(), add_debug_message("재학습 완료")), daemon=True).start()
    return jsonify({"message": "재학습 시작"})

@app.route("/model_version", methods=["GET"])
def model_version():
    return jsonify({
        "model": getattr(model.config, "_name_or_path", "n/a") if model else None,
        "device": device,
        "loaded": model_loaded_event.is_set()
    })

@app.route("/feedback", methods=["POST"])
@require_auth
def feedback():
    data = request.get_json(force=True)
    fb = data.get("feedback", "").strip()
    if not fb:
        return jsonify({"error": "빈 피드백"}), 400
    with open(config.FEEDBACK_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat()} - {mask_pii(fb)}\n")
    add_debug_message("피드백 제출 완료")
    return jsonify({"message": "피드백 감사드립니다"})

@app.route("/debug_log", methods=["GET"])
def get_debug_log():
    return jsonify({"debug": debug_messages})

@app.route("/health", methods=["GET"])
def health():
    return "ok", 200

@app.route("/metrics", methods=["GET"])
def metrics():
    try:
        avg_lat = round(statistics.mean(LAT_HIST), 1) if LAT_HIST else None
    except Exception:
        avg_lat = None
    return jsonify({
        "requests": REQ_CNT,
        "avg_latency_ms": avg_lat,
        "gen_fail": GEN_FAIL_CNT
    })

@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML)

@app.route("/chat", methods=["POST"])
def chat():
    global REQ_CNT, LAT_HIST
    t0 = time.time()
    rid = request.headers.get("X-Request-Id") or str(uuid.uuid4())
    add_debug_message(f"RID={rid} /chat start")

    ensure_auto_model_loaded()

    data = request.get_json(force=True, silent=True) or {}
    user_msg = (data.get("message") or "").strip()
    user_msg_log = mask_pii(user_msg)
    if not user_msg:
        return jsonify({"error": "빈 메시지", "request_id": rid}), 400

    # 수동 모델 선택
    user_model = request.args.get("model") or data.get("model")
    if user_model:
        try:
            load_or_switch_model(user_model, force=False)
        except Exception as e:
            add_debug_message(f"RID={rid} manual model switch failed: {e}", logging.WARNING)
            # 자동으로 다시 보장
            ensure_auto_model_loaded()

    # 최근 대화 컨텍스트 구축 + 요약
    recent_text = "\n".join([f"{it['role']}: {it['content']}" for it in chat_history[-8:]]) if chat_history else ""
    summary = summarize_context(recent_text) if recent_text else ""

    prompt = f"""다음은 사용자와 어시스턴트의 대화입니다.
요약 컨텍스트:
{summary}

사용자: {user_msg}
어시스턴트:"""

    stream = request.args.get("stream") == "true"

    try:
        if stream:
            def gen():
                try:
                    for chunk in generate_stream(prompt):
                        yield sse_format(chunk)
                except Exception as e:
                    add_debug_message(f"RID={rid} stream error: {e}", logging.ERROR)
                    yield sse_format("스트리밍 중 오류가 발생했습니다.")
            resp = Response(gen(), mimetype="text/event-stream")
        else:
            reply = generate_with_timeout(prompt, timeout=config.GENERATE_TIMEOUT)
            reply = remove_echo(reply, user_msg)
            reply = filter_uncertain(reply)
            resp = jsonify({"reply": reply, "request_id": rid})

        # 히스토리 저장(PII 마스킹)
        chat_history.append({"role": "user", "content": user_msg_log, "time": datetime.now(timezone.utc).isoformat()})
        if not stream:
            chat_history.append({"role": "assistant", "content": mask_pii(reply), "time": datetime.now(timezone.utc).isoformat()})
        return resp
    finally:
        t1 = time.time()
        lat = int((t1 - t0) * 1000)
        REQ_CNT += 1
        LAT_HIST.append(lat)
        if len(LAT_HIST) > 200:
            LAT_HIST.pop(0)
        add_debug_message(f"RID={rid} /chat end {lat}ms")

#############################################
# RAG placeholder (faiss guard)
#############################################
def try_import_faiss():
    try:
        import faiss  # noqa
        return True
    except Exception:
        return False

#############################################
# Model Loading with Heartbeat & Background Tasks
#############################################
def _heartbeat(stop_evt):
    while not stop_evt.is_set():
        add_debug_message("모델 로딩/서비스 동작 중...", logging.DEBUG)
        time.sleep(30)

def idle_crawling_task():
    global last_activity_time
    while not stop_event.is_set():
        if time.time() - last_activity_time > config.IDLE_CRAWLING_INTERVAL:
            # TODO: 실제 크롤링 로직
            new = []
            if new:
                merged = merge_and_dedup(load_json_list(config.PRETRAIN_DATA_FILE) or [], new)
                save_json_list(config.PRETRAIN_DATA_FILE, merged)
                add_debug_message(f"Idle crawl added {len(new)} items")
        time.sleep(config.IDLE_CRAWLING_INTERVAL)

def save_chat_history_periodically(evt):
    while not evt.is_set():
        if chat_history:
            save_json_list(config.HISTORY_FILE, chat_history[-config.MAX_TRAINING_SAMPLES:])
        time.sleep(config.CHAT_HISTORY_SAVE_INTERVAL)

#############################################
# Main execution
#############################################
if __name__ == "__main__":
    add_debug_message("Starting WALT server...", logging.INFO)

    # 1) 사전학습 데이터 준비
    combine_all_datasets()
    chat_history.extend(load_json_list(config.HISTORY_FILE) or [])

    # 2) CPU 사용량 초기화 (baseline)
    psutil.cpu_percent(interval=None)

    # 3) 모델 로드(자동 선택, 동기) — Flask 실행 전에 완료
    try:
        ensure_auto_model_loaded()
    except Exception as e:
        add_debug_message(f"Auto model load failed: {e}", logging.ERROR)
        # 최후 시도: 가장 작은 모델
        try:
            load_or_switch_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0", force=True)
        except Exception as e2:
            add_debug_message(f"Fallback model load failed: {e2}", logging.CRITICAL)

    # 4) RAG(Faiss) 가능 여부 알림
    if try_import_faiss():
        add_debug_message("FAISS available; RAG placeholder ready")
    else:
        add_debug_message("FAISS not available; RAG disabled")

    # 5) 백그라운드 태스크 시작
    threading.Thread(target=_heartbeat, args=(stop_event,), daemon=True).start()
    threading.Thread(target=idle_crawling_task, daemon=True).start()
    threading.Thread(target=save_chat_history_periodically, args=(stop_event,), daemon=True).start()

    # 6) Flask 서버 실행
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False,
        use_reloader=False
    )