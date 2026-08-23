from __future__ import annotations  # allows modern type hint syntax on older Python

import json  # for reading and writing JSON files
from datetime import datetime, timezone  # for getting the current timestamp
from pathlib import Path  # for working with file system paths
from uuid import uuid4  # for stable ids on query and follow-up log entries


# ── File Paths ────────────────────────────────────────────────────────────────
STORE = Path(__file__).resolve().parents[2] / "store"
# __file__ = this file (logger.py) inside platform/app/services/
# .resolve() = make the path absolute (no relative ".." confusion)
# .parents[2] = go up 2 levels: services → app → platform
# / "store" = join the store folder → platform/store/

LOGS_PATH = STORE / "logs.json"
# path to platform/store/logs.json — where all query logs are stored

PERF_PATH = STORE / "model_performance.json"
# path to platform/store/model_performance.json — where model performance data is stored


def ensure_store_files() -> None:
    # makes sure both JSON files exist before we try to read/write them
    # called once at server startup so we never get "file not found" errors

    STORE.mkdir(parents=True, exist_ok=True)
    # creates the store/ directory if it doesn't exist
    # parents=True: also creates any missing parent directories
    # exist_ok=True: don't raise an error if the directory already exists

    if not LOGS_PATH.exists():
        LOGS_PATH.write_text("[]")
    # if logs.json doesn't exist, create it with an empty JSON array "[]"
    # [] = empty list in JSON — logs are stored as an array of objects

    if not PERF_PATH.exists():
        PERF_PATH.write_text("{}")
    # if model_performance.json doesn't exist, create it with an empty JSON object "{}"
    # {} = empty dict in JSON — performance data is stored as {domain: {stats}}


def read_logs() -> list[dict]:
    ensure_store_files()

    try:
        current = json.loads(LOGS_PATH.read_text())
        if isinstance(current, list):
            return current
    except Exception:
        pass

    return []


def append_log(entry: dict) -> dict:
    # adds a new query result to the logs.json file
    # each call to this function adds ONE entry to the array in the file

    ensure_store_files()  # make sure the files exist before trying to read/write

    current = read_logs()

    saved_entry = {
        **entry,  # spread all key-value pairs from the entry dict
        "id": entry.get("id") or str(uuid4()),
        "timestamp": entry.get("timestamp") or datetime.now(timezone.utc).isoformat()
        # add a timestamp to every log entry
        # datetime.now(timezone.utc) = current time in UTC timezone
        # .isoformat() converts to string like "2026-05-16T11:30:00+00:00"
    }

    current.append(saved_entry)

    LOGS_PATH.write_text(json.dumps(current, indent=2))
    # json.dumps(current) = convert the whole list back to a JSON string
    # indent=2 = pretty-print with 2-space indentation (human-readable)
    # .write_text() = overwrite the file with the updated content

    return saved_entry
