#!/usr/bin/env python3
"""Seed routing clusters so adaptive routing can skip full exploration.

Usage:
  cd platform
  ../.venv/bin/python scripts/seed_routing.py bootstrap          # fast (~30s), no model calls
  ../.venv/bin/python scripts/seed_routing.py bootstrap --cluster python
  ../.venv/bin/python scripts/seed_routing.py verify             # check direct routing works
  ../.venv/bin/python scripts/seed_routing.py status             # show cluster readiness
  ../.venv/bin/python scripts/seed_routing.py live --max 3       # slow, real model calls
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT.parent / ".env")

from app.services.adaptive_routing.seed import (  # noqa: E402
    bootstrap_all_clusters,
    bootstrap_cluster,
    get_seed_status,
    load_seed_queries,
    live_seed_all,
    verify_direct_routing,
)


def cmd_status(_: argparse.Namespace) -> int:
    status = get_seed_status()
    print(json.dumps(status, indent=2))
    return 0


def cmd_bootstrap(args: argparse.Namespace) -> int:
    queries = load_seed_queries()
    if args.cluster:
        if args.cluster not in queries:
            print(f"Unknown cluster: {args.cluster}", file=sys.stderr)
            return 1
        batch = queries[args.cluster][: args.limit] if args.limit else queries[args.cluster]
        result = bootstrap_cluster(args.cluster, batch)
        print(f"Bootstrapped {args.cluster}: count={result['count']} mode={result['mode']}")
    else:
        if args.limit:
            queries = {k: v[: args.limit] for k, v in queries.items()}
        results = bootstrap_all_clusters(queries)
        for cid, row in results.items():
            print(f"  {cid}: count={row['count']} mode={row['mode']} best={row['best_model']}")
        print(f"\nBootstrapped {len(results)} clusters.")
    status = get_seed_status()
    print(f"\nAll clusters ready for direct routing: {status['all_ready']}")
    return 0


def cmd_verify(_: argparse.Namespace) -> int:
    results = verify_direct_routing()
    ok = 0
    for cid, row in results.items():
        mark = "OK" if row["direct_routing_worked"] else "NEEDS SEED"
        print(f"[{mark}] {cid}: mode={row['mode']} models={row['model_count']} reason={row['reason']}")
        if row["direct_routing_worked"]:
            ok += 1
    print(f"\n{ok}/{len(results)} clusters routing directly.")
    return 0 if ok == len(results) else 1


def cmd_live(args: argparse.Namespace) -> int:
    print("Running live seed (calls all models per query — this is slow)...")
    outcomes = live_seed_all(max_per_cluster=args.max)
    for cid, rows in outcomes.items():
        print(f"  {cid}: {len(rows)} queries seeded")
    status = get_seed_status()
    print(f"\nAll clusters ready: {status['all_ready']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed adaptive routing clusters")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Show cluster readiness").set_defaults(func=cmd_status)

    p_boot = sub.add_parser("bootstrap", help="Fast seed via embeddings (no model calls)")
    p_boot.add_argument("--cluster", help="Only seed one cluster (python, chemistry, gita, general)")
    p_boot.add_argument("--limit", type=int, help="Max queries per cluster (default: all 22)")
    p_boot.set_defaults(func=cmd_bootstrap)

    sub.add_parser("verify", help="Verify direct routing after seeding").set_defaults(func=cmd_verify)

    p_live = sub.add_parser("live", help="Slow seed via real full exploration")
    p_live.add_argument("--max", type=int, default=5, help="Max queries per cluster (default 5)")
    p_live.set_defaults(func=cmd_live)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
