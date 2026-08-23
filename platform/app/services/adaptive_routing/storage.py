"""JSON persistence for routing table and routing decision logs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.services.adaptive_routing.config import SEED_CLUSTERS

STORE = Path(__file__).resolve().parents[3] / "store"
ROUTING_TABLE_PATH = STORE / "routing_table.json"
ROUTING_LOGS_PATH = STORE / "routing_logs.json"


def _read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _write_json(path: Path, data) -> None:
    STORE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def _empty_cluster(cluster_id: str, seed_domain: str | None, parent_id: str | None) -> dict:
    default_model = SEED_CLUSTERS.get(seed_domain or cluster_id, {}).get("default_model", "groq_model")
    return {
        "id": cluster_id,
        "parent_id": parent_id,
        "seed_domain": seed_domain or cluster_id.split("/")[0],
        "best_model": default_model,
        "avg_accuracy": 0.0,
        "avg_semantic_accuracy": 0.0,
        "avg_confidence": 0.0,
        "avg_latency": 0.0,
        "count": 0,
        "exploration_count": 0,
        "queries_since_exploration": 0,
        "last_full_exploration_at": None,
        "mode": "exploration",
        "force_explore": False,
        "centroid": None,
        "recent_routed_metrics": [],
    }


def ensure_routing_files() -> None:
    STORE.mkdir(parents=True, exist_ok=True)
    if not ROUTING_TABLE_PATH.exists():
        clusters = {
            cluster_id: _empty_cluster(cluster_id, cluster_id, None)
            for cluster_id in SEED_CLUSTERS
        }
        _write_json(
            ROUTING_TABLE_PATH,
            {"clusters": clusters, "next_subcluster_index": {k: 1 for k in SEED_CLUSTERS}},
        )
    if not ROUTING_LOGS_PATH.exists():
        _write_json(ROUTING_LOGS_PATH, [])


def load_routing_table() -> dict:
    ensure_routing_files()
    data = _read_json(ROUTING_TABLE_PATH, {"clusters": {}, "next_subcluster_index": {}})
    if "clusters" not in data:
        data["clusters"] = {}
    if "next_subcluster_index" not in data:
        data["next_subcluster_index"] = {k: 1 for k in SEED_CLUSTERS}
    for cluster_id in SEED_CLUSTERS:
        if cluster_id not in data["clusters"]:
            data["clusters"][cluster_id] = _empty_cluster(cluster_id, cluster_id, None)
    return data


def save_routing_table(data: dict) -> None:
    ensure_routing_files()
    _write_json(ROUTING_TABLE_PATH, data)


def get_cluster(cluster_id: str) -> dict | None:
    table = load_routing_table()
    return table["clusters"].get(cluster_id)


def update_cluster(cluster_id: str, updates: dict) -> dict:
    table = load_routing_table()
    cluster = table["clusters"].get(cluster_id)
    if cluster is None:
        parent = cluster_id.split("/")[0] if "/" in cluster_id else None
        seed = parent or cluster_id
        cluster = _empty_cluster(cluster_id, seed, parent if parent != cluster_id else None)
        table["clusters"][cluster_id] = cluster
    cluster.update(updates)
    table["clusters"][cluster_id] = cluster
    save_routing_table(table)
    return cluster


def append_routing_log(entry: dict) -> dict:
    ensure_routing_files()
    logs = _read_json(ROUTING_LOGS_PATH, [])
    saved = {
        **entry,
        "id": entry.get("id") or str(uuid4()),
        "timestamp": entry.get("timestamp") or datetime.now(timezone.utc).isoformat(),
    }
    logs.append(saved)
    _write_json(ROUTING_LOGS_PATH, logs)
    return saved


def read_routing_logs() -> list[dict]:
    ensure_routing_files()
    return _read_json(ROUTING_LOGS_PATH, [])


def allocate_subcluster_id(parent_id: str) -> str:
    table = load_routing_table()
    idx_map = table.setdefault("next_subcluster_index", {})
    idx = int(idx_map.get(parent_id, 1))
    sub_id = f"{parent_id}/sub_{idx}"
    idx_map[parent_id] = idx + 1
    save_routing_table(table)
    return sub_id
