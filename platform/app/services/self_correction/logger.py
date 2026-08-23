"""JSON logging for self-correction loop runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

STORE = Path(__file__).resolve().parents[2] / "store"
SELF_CORRECTION_LOGS_PATH = STORE / "self_correction_logs.json"


def ensure_self_correction_logs() -> None:
    STORE.mkdir(parents=True, exist_ok=True)
    if not SELF_CORRECTION_LOGS_PATH.exists():
        SELF_CORRECTION_LOGS_PATH.write_text("[]")


def read_self_correction_logs() -> list[dict]:
    ensure_self_correction_logs()
    try:
        data = json.loads(SELF_CORRECTION_LOGS_PATH.read_text())
        return data if isinstance(data, list) else []
    except Exception:
        return []


def append_self_correction_log(entry: dict) -> dict:
    ensure_self_correction_logs()
    logs = read_self_correction_logs()
    saved = {
        **entry,
        "id": entry.get("id") or str(uuid4()),
        "timestamp": entry.get("timestamp") or datetime.now(timezone.utc).isoformat(),
    }
    logs.append(saved)
    SELF_CORRECTION_LOGS_PATH.write_text(json.dumps(logs, indent=2))
    return saved
