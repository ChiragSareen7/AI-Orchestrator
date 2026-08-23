"""Tests for the adaptive query routing layer."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.services.adaptive_routing import config as routing_config
from app.services.adaptive_routing import storage as routing_storage
from app.services.adaptive_routing.classifier import classify_query
from app.services.adaptive_routing.reexploration import force_reexploration, should_full_explore
from app.services.adaptive_routing.router import plan_query_execution, record_query_outcome


class AdaptiveRoutingTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        self.store = Path(self._tmpdir)
        routing_storage.STORE = self.store
        routing_storage.ROUTING_TABLE_PATH = self.store / "routing_table.json"
        routing_storage.ROUTING_LOGS_PATH = self.store / "routing_logs.json"
        routing_storage.ensure_routing_files()

        self._orig_min_samples = routing_config.MIN_SAMPLES_FOR_DIRECT_ROUTING
        self._orig_periodic = routing_config.PERIODIC_REEXPLORATION_INTERVAL
        self._orig_acc_drop = routing_config.ACCURACY_DROP_THRESHOLD
        self._orig_lat_inc = routing_config.LATENCY_INCREASE_THRESHOLD_MS
        routing_config.MIN_SAMPLES_FOR_DIRECT_ROUTING = 2
        routing_config.PERIODIC_REEXPLORATION_INTERVAL = 100
        routing_config.ACCURACY_DROP_THRESHOLD = 0.15
        routing_config.LATENCY_INCREASE_THRESHOLD_MS = 500

    def tearDown(self) -> None:
        routing_config.MIN_SAMPLES_FOR_DIRECT_ROUTING = self._orig_min_samples
        routing_config.PERIODIC_REEXPLORATION_INTERVAL = self._orig_periodic
        routing_config.ACCURACY_DROP_THRESHOLD = self._orig_acc_drop
        routing_config.LATENCY_INCREASE_THRESHOLD_MS = self._orig_lat_inc
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _analysis(self, domain: str = "python") -> dict:
        return {
            "domain": domain,
            "complexity": "low",
            "intent": "fact_lookup",
            "domain_scores": {domain: 1},
        }

    def _best_response(self, model: str = "python_model", accuracy: float = 0.9, latency: float = 100.0) -> dict:
        return {
            "model": model,
            "prompt_version": "v1",
            "metrics": {
                "accuracyScore": accuracy,
                "confidenceScore": accuracy - 0.05,
                "latency": latency,
                "tokenUsage": 10,
                "cost": 0.0,
                "errorRate": 0.0,
            },
        }

    def test_unseen_query_triggers_full_exploration(self) -> None:
        with patch(
            "app.services.adaptive_routing.router.classify_query",
            return_value=type(
                "R",
                (),
                {
                    "cluster_id": "python",
                    "method": "llm_fallback",
                    "confidence": 0.75,
                    "similarities": {},
                    "ambiguous": True,
                    "llm_domain": "python",
                },
            )(),
        ), patch(
            "app.services.adaptive_routing.router.compute_query_embedding",
            return_value=[0.11] * 384,
        ):
            plan = plan_query_execution("What is a python decorator?", self._analysis("python"))

        self.assertEqual(plan["mode"], "full_exploration")
        self.assertEqual(len(plan["models"]), 4)
        self.assertEqual(plan["routing_decision"]["reason"], "insufficient_cluster_samples")

        logs = routing_storage.read_routing_logs()
        self.assertTrue(any(entry.get("event") == "routing_decision" for entry in logs))

    def test_confident_cluster_routes_directly_to_best_model(self) -> None:
        routing_storage.update_cluster(
            "python",
            {
                "count": 5,
                "exploration_count": 5,
                "mode": "direct",
                "best_model": "python_model",
                "avg_accuracy": 0.85,
                "avg_semantic_accuracy": 0.85,
                "avg_confidence": 0.8,
                "avg_latency": 120.0,
                "queries_since_exploration": 0,
                "centroid": [0.1] * 384,
            },
        )

        with patch("app.services.adaptive_routing.router.MIN_SAMPLES_FOR_DIRECT_ROUTING", 2), patch(
            "app.services.adaptive_routing.router.classify_query",
            return_value=type(
                "R",
                (),
                {
                    "cluster_id": "python",
                    "method": "embedding_match",
                    "confidence": 0.9,
                    "similarities": {"python": 0.9},
                    "ambiguous": False,
                    "llm_domain": None,
                },
            )(),
        ), patch(
            "app.services.adaptive_routing.router.compute_query_embedding",
            return_value=[0.1] * 384,
        ):
            plan = plan_query_execution("Explain python list comprehension", self._analysis("python"))

        self.assertEqual(plan["mode"], "direct")
        self.assertEqual(plan["models"], ["python_model"])
        self.assertEqual(plan["routing_decision"]["reason"], "confident_cluster_direct_route")

    def test_metric_drop_triggers_re_exploration(self) -> None:
        cluster = routing_storage.update_cluster(
            "python",
            {
                "count": 10,
                "exploration_count": 5,
                "mode": "direct",
                "best_model": "python_model",
                "avg_accuracy": 0.85,
                "avg_semantic_accuracy": 0.85,
                "avg_confidence": 0.8,
                "avg_latency": 120.0,
                "queries_since_exploration": 1,
                "centroid": [0.2] * 384,
            },
        )
        bad_metrics = {
            "accuracyScore": 0.5,
            "confidenceScore": 0.45,
            "latency": 900.0,
            "tokenUsage": 10,
            "cost": 0.0,
            "errorRate": 0.0,
        }
        with patch("app.services.adaptive_routing.reexploration.MIN_SAMPLES_FOR_DIRECT_ROUTING", 2):
            forced, reasons = should_full_explore(cluster, bad_metrics)
        self.assertTrue(forced)
        self.assertTrue(any("accuracy_drop" in r or "confidence_drop" in r or "latency_increase" in r for r in reasons))

        plan = {
            "cluster_id": "python",
            "query_embedding": [0.2] * 384,
            "mode": "direct",
            "routing_decision": {"cluster_id": "python", "mode": "direct"},
        }
        with patch("app.services.adaptive_routing.reexploration.MIN_SAMPLES_FOR_DIRECT_ROUTING", 2):
            record_query_outcome(
                plan,
                "python tuple vs list",
                [{"model": "python_model", "metrics": bad_metrics}],
                {"model": "python_model", "metrics": bad_metrics},
                self._analysis("python"),
            )

        updated = routing_storage.get_cluster("python")
        assert updated is not None
        self.assertEqual(updated["mode"], "exploration")
        self.assertTrue(updated["force_explore"])

        with patch("app.services.adaptive_routing.router.MIN_SAMPLES_FOR_DIRECT_ROUTING", 2), patch(
            "app.services.adaptive_routing.router.classify_query",
            return_value=type(
                "R",
                (),
                {
                    "cluster_id": "python",
                    "method": "embedding_match",
                    "confidence": 0.9,
                    "similarities": {"python": 0.9},
                    "ambiguous": False,
                    "llm_domain": None,
                },
            )(),
        ), patch(
            "app.services.adaptive_routing.router.compute_query_embedding",
            return_value=[0.2] * 384,
        ):
            next_plan = plan_query_execution("python tuple vs list", self._analysis("python"))

        self.assertEqual(next_plan["mode"], "full_exploration")
        self.assertIn("re_exploration_triggered", next_plan["routing_decision"]["reason"])

    def test_manual_override_force_explore(self) -> None:
        routing_storage.update_cluster(
            "python",
            {
                "count": 10,
                "mode": "direct",
                "best_model": "python_model",
                "avg_accuracy": 0.9,
                "avg_confidence": 0.9,
                "avg_latency": 100.0,
                "centroid": [0.3] * 384,
            },
        )

        result = force_reexploration(cluster_id="python")
        self.assertEqual(result["updated_clusters"], ["python"])

        cluster = routing_storage.get_cluster("python")
        assert cluster is not None
        self.assertTrue(cluster["force_explore"])
        self.assertEqual(cluster["mode"], "exploration")

        with patch(
            "app.services.adaptive_routing.router.classify_query",
            return_value=type(
                "R",
                (),
                {
                    "cluster_id": "python",
                    "method": "embedding_match",
                    "confidence": 0.95,
                    "similarities": {"python": 0.95},
                    "ambiguous": False,
                    "llm_domain": None,
                },
            )(),
        ), patch(
            "app.services.adaptive_routing.router.compute_query_embedding",
            return_value=[0.3] * 384,
        ):
            plan = plan_query_execution(
                "How do python decorators work?",
                self._analysis("python"),
                force_explore_cluster="python",
            )

        self.assertEqual(plan["mode"], "full_exploration")
        self.assertTrue(
            any("request_force_explore_cluster" in r for r in plan["routing_decision"]["reexploration_reasons"])
        )

    def test_record_query_outcome_updates_table_and_logs(self) -> None:
        plan = {
            "cluster_id": "python",
            "query_embedding": [0.4] * 384,
            "mode": "full_exploration",
            "routing_decision": {"cluster_id": "python", "mode": "full_exploration"},
        }
        responses = [{"model": "python_model", "metrics": self._best_response()["metrics"]}]
        record_query_outcome(plan, "python dict comprehension", responses, self._best_response(), self._analysis())

        cluster = routing_storage.get_cluster("python")
        assert cluster is not None
        self.assertEqual(cluster["best_model"], "python_model")
        self.assertEqual(cluster["count"], 1)
        self.assertIsNotNone(cluster["centroid"])

        logs = routing_storage.read_routing_logs()
        self.assertTrue(any(entry.get("event") == "cluster_state_updated" for entry in logs))


if __name__ == "__main__":
    unittest.main()
