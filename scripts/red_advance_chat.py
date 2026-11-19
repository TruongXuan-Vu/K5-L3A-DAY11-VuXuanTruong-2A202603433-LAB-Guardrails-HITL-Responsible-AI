#!/usr/bin/env python3
"""
Chatbox test Red Advance (Flask, localhost).

Cho các bạn trong lớp gõ prompt tấn công vào Red Advance và thấy ngay
kết quả có leak secret hay không. Dùng CHÍNH logic chấm của lab
(attacks.classify_attack_outcome) để quyết định leaked/layer — không tự
viết lại detection.

Chạy từ gốc repo:
    python scripts/red_advance_chat.py

Mở http://127.0.0.1:8000 trong trình duyệt.
Mỗi lượt được append vào outputs/red_advance_chat_log.jsonl.
"""
from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, render_template, request

# Cho phép import agents.*, core.*, attacks.* (giống src/main.py).
_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Import sau khi đã sửa sys.path. core.config tự load .env lúc import.
from agents.guards_agent import create_red_agent_advance  # noqa: E402
from attacks.attacks import classify_attack_outcome  # noqa: E402
from core.utils import chat_with_agent  # noqa: E402

LOG_PATH = _ROOT / "outputs" / "red_advance_chat_log.jsonl"

app = Flask(__name__, template_folder=str(_SCRIPT_DIR / "templates"))

# Khởi tạo 1 lần lúc startup. .env đã được load sẵn bởi core.config.
try:
    AGENT, RUNNER = create_red_agent_advance()
    INIT_ERROR: str | None = None
except Exception as exc:  # thiếu key / NeMo / cấu hình
    AGENT = RUNNER = None
    INIT_ERROR = f"{type(exc).__name__}: {exc}"
    print(f"[red_advance_chat] Không khởi tạo được Red Advance: {INIT_ERROR}")
    print("[red_advance_chat] Kiểm tra .env (OPENAI_API_KEY) và nemoguardrails.")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_log() -> list[dict]:
    if not LOG_PATH.is_file():
        return []
    rows: list[dict] = []
    for line in LOG_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _append_log(row: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


@app.route("/")
def index():
    return render_template("red_advance_chat.html")


@app.route("/results")
def results():
    rows = _read_log()
    leaks = sum(1 for r in rows if r.get("leaked"))
    return jsonify({"rows": rows, "total": len(rows), "leaks": leaks})


@app.route("/chat", methods=["POST"])
def chat():
    if INIT_ERROR or AGENT is None or RUNNER is None:
        return jsonify({
            "error": f"Red Advance chưa sẵn sàng: {INIT_ERROR}",
            "leaked": False,
            "layer": "init_error",
            "blocked_at": "SERVER ERROR — agent không khởi tạo được",
        }), 503

    data = request.get_json(silent=True) or {}
    prompt = str(data.get("prompt", "")).strip()
    if not prompt:
        return jsonify({"error": "Prompt rỗng", "leaked": False}), 400

    try:
        response_text, _ = asyncio.run(
            chat_with_agent(AGENT, RUNNER, prompt)
        )
        outcome = classify_attack_outcome(
            prompt, response_text, target_name="red_advance"
        )
        error = None
    except Exception as exc:  # vd openai.RateLimitError 429
        response_text = ""
        outcome = {
            "leaked": False,
            "layer": "error",
            "blocked_at": f"ERROR — {type(exc).__name__}",
        }
        error = f"{type(exc).__name__}: {exc}"

    row = {
        "timestamp": _now_iso(),
        "prompt": prompt,
        "response": response_text,
        "leaked": bool(outcome.get("leaked")),
        "layer": outcome.get("layer"),
        "blocked_at": outcome.get("blocked_at"),
        "error": error,
    }
    _append_log(row)

    payload = dict(row)
    payload["response_preview"] = response_text[:500]
    return jsonify(payload)


if __name__ == "__main__":
    port = 8000
    print("=" * 60)
    print("Red Advance Chatbox")
    print(f"Open http://127.0.0.1:{port}")
    print(f"Log  -> {LOG_PATH}")
    if INIT_ERROR:
        print(f"WARNING: {INIT_ERROR}")
    print("=" * 60)
    app.run(host="127.0.0.1", port=port, debug=False)
