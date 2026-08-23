from __future__ import annotations  # allows modern type hint syntax on older Python

import json        # for reading and writing JSON files
from pathlib import Path  # for working with file paths


# ── File Paths ────────────────────────────────────────────────────────────────
STORE = Path(__file__).resolve().parents[2] / "store"
# __file__ = this file (learning.py), e.g. platform/app/services/learning.py
# .resolve() = make the path absolute
# .parents[2] = go up 2 levels: services/ → app/ → platform/ (that's index 0, 1, 2)
# Wait: parents[0]=services, parents[1]=app, parents[2]=platform → so this is platform/store
# / "store" joins the store/ subfolder

PERF_PATH = STORE / "model_performance.json"
# full path to the performance file: platform/store/model_performance.json
# structure: {"chemistry": {"best_model": "organic_model", "avg_accuracy": 0.84, ...}}

LOGS_PATH = STORE / "logs.json"
# full path to the logs file: platform/store/logs.json
# structure: [{query, analysis, responses, best_model, metrics, ...}, ...]


def _read_json(path: Path, default):
    # helper to safely read a JSON file
    if not path.exists():
        return default  # file doesn't exist yet → return the default value
    try:
        return json.loads(path.read_text())
        # path.read_text() reads the entire file as a string
        # json.loads() parses the JSON string into a Python dict or list
    except Exception:
        return default  # file is corrupt/empty → return the default value


def _write_json(path: Path, data) -> None:
    # helper to write Python data to a JSON file
    path.write_text(json.dumps(data, indent=2))
    # json.dumps() converts Python dict/list to a JSON string
    # indent=2 makes the JSON file human-readable with 2-space indentation
    # path.write_text() writes the string to the file (overwrites existing content)


def get_domain_bias(domain: str) -> str | None:
    # reads model_performance.json and returns the best model for this domain
    # this is called by model_router.py to influence which model gets tried first
    perf = _read_json(PERF_PATH, {})  # read the performance data; default to empty dict if file missing
    d = perf.get(domain, {})          # get this domain's data; empty dict if domain not seen yet
    return d.get("best_model")        # return the best_model value, or None if not set yet


def update_learning(domain: str, best_model: str, metrics: dict) -> None:
    # after every query, update the running performance statistics for the winning model
    # this is the "feedback loop" that makes the system learn over time

    perf = _read_json(PERF_PATH, {})  # read current performance data

    row = perf.get(domain, {
        # get this domain's existing data, or create a fresh entry if this is the first query
        "best_model": best_model,   # initial best model
        "avg_accuracy": 0.0,        # running average accuracy
        "avg_latency": 0.0,         # running average latency
        "count": 0,                 # how many queries have been processed for this domain
    })

    count = int(row.get("count", 0)) + 1  # increment the query count by 1

    # decide which accuracy value to use: prefer semantic accuracy if available
    sem = metrics.get("semantic")  # get the semantic evaluation sub-dict
    if isinstance(sem, dict) and not (sem.get("metadata") or {}).get("error"):
        # semantic dict exists AND doesn't have an error → use semantic accuracy (more reliable)
        acc_val = float(sem.get("accuracy", metrics["accuracyScore"]))
        # sem.get("accuracy", metrics["accuracyScore"]) = use semantic accuracy, fall back to lexical
    else:
        acc_val = metrics["accuracyScore"]  # use the lexical accuracy score

    # update the running average using the "incremental mean" formula:
    # new_avg = (old_avg * (count - 1) + new_value) / count
    row["avg_accuracy"] = round(
        ((row.get("avg_accuracy", 0.0) * (count - 1)) + acc_val) / count, 4
    )
    # this formula avoids storing all historical values — just the count and running average

    row["avg_latency"] = round(
        ((row.get("avg_latency", 0.0) * (count - 1)) + metrics["latency"]) / count, 2
    )
    # same incremental average formula for latency

    row["best_model"] = best_model  # whoever won THIS query becomes the "best model" for next time
    row["count"] = count            # save the updated count

    perf[domain] = row  # put the updated row back into the performance dict
    _write_json(PERF_PATH, perf)  # save to disk


def generate_report() -> dict:
    # creates a summary of the system's overall performance across all domains
    # called by the GET /report endpoint

    perf = _read_json(PERF_PATH, {})  # read model performance data
    logs = _read_json(LOGS_PATH, [])  # read all query logs

    total = len(logs)  # total number of queries processed since the system started
    avg_latency = 0.0
    if total:
        # compute average latency across all logged queries
        avg_latency = round(
            sum(l.get("metrics", {}).get("latency", 0.0) for l in logs) / total, 2
        )
        # sum() adds up all latency values from all log entries
        # divide by total to get the average

    return {
        "best_model_per_domain": {k: v.get("best_model") for k, v in perf.items()},
        "average_latency": avg_latency,
        "accuracy_comparison": {k: v.get("avg_accuracy", 0.0) for k, v in perf.items()},
        "total_queries": total,
        "domain_stats": {
            k: {
                "best_model": v.get("best_model"),
                "avg_accuracy": v.get("avg_accuracy", 0.0),
                "avg_latency": v.get("avg_latency", 0.0),
                "count": v.get("count", 0),
            }
            for k, v in perf.items()
        },
    }
