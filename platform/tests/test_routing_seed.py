"""Tests for routing seed + direct routing verification."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.services.adaptive_routing import config as routing_config
from app.services.adaptive_routing import storage as routing_storage
from app.services.adaptive_routing.seed import (
    bootstrap_all_clusters,
    bootstrap_cluster,
    get_seed_status,
    verify_direct_routing,
)


class RoutingSeedTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        self.store = Path(self._tmpdir)
        routing_storage.STORE = self.store
        routing_storage.ROUTING_TABLE_PATH = self.store / "routing_table.json"
        routing_storage.ROUTING_LOGS_PATH = self.store / "routing_logs.json"
        routing_storage.ensure_routing_files()

        self._orig_min = routing_config.MIN_SAMPLES_FOR_DIRECT_ROUTING
        routing_config.MIN_SAMPLES_FOR_DIRECT_ROUTING = 5

        self._sample_queries = {
            "python": [f"Python question about lists number {i}" for i in range(5)],
            "chemistry": [f"Chemistry benzene boiling point question {i}" for i in range(5)],
            "gita": [f"Gita krishna dharma verse question {i}" for i in range(5)],
            "general": [f"General knowledge capital city question {i}" for i in range(5)],
        }

    def tearDown(self) -> None:
        routing_config.MIN_SAMPLES_FOR_DIRECT_ROUTING = self._orig_min
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_bootstrap_cluster_reaches_direct_mode(self) -> None:
        with patch(
            "app.services.adaptive_routing.seed.compute_query_embedding",
            return_value=[0.1] * 384,
        ), patch("app.services.adaptive_routing.config.MIN_SAMPLES_FOR_DIRECT_ROUTING", 5):
            result = bootstrap_cluster("python", self._sample_queries["python"])

        self.assertEqual(result["count"], 5)
        self.assertEqual(result["mode"], "direct")
        self.assertEqual(result["best_model"], "python_model")
        self.assertIsNotNone(result["centroid"])

    def test_bootstrap_all_clusters(self) -> None:
        with patch(
            "app.services.adaptive_routing.seed.compute_query_embedding",
            return_value=[0.2] * 384,
        ), patch("app.services.adaptive_routing.config.MIN_SAMPLES_FOR_DIRECT_ROUTING", 5):
            results = bootstrap_all_clusters(self._sample_queries)

        self.assertEqual(len(results), 4)
        status = get_seed_status()
        self.assertTrue(status["all_ready"])

    def test_verify_direct_routing_after_bootstrap(self) -> None:
        with patch(
            "app.services.adaptive_routing.seed.compute_query_embedding",
            return_value=[0.3] * 384,
        ), patch("app.services.adaptive_routing.config.MIN_SAMPLES_FOR_DIRECT_ROUTING", 5), patch(
            "app.services.adaptive_routing.router.compute_query_embedding",
            return_value=[0.3] * 384,
        ), patch("app.services.adaptive_routing.router.MIN_SAMPLES_FOR_DIRECT_ROUTING", 5):
            bootstrap_all_clusters(self._sample_queries)

            results = verify_direct_routing(
                {
                    "python": "Explain python decorators",
                    "chemistry": "What is benzene boiling point",
                    "gita": "What does Krishna say about karma",
                    "general": "What is the capital of Italy",
                }
            )

        for cid, row in results.items():
            self.assertEqual(row["mode"], "direct", f"{cid} should direct route")
            self.assertEqual(row["model_count"], 1, f"{cid} should call 1 model")
            self.assertTrue(row["direct_routing_worked"])

    def test_unseeded_cluster_still_explores(self) -> None:
        with patch(
            "app.services.adaptive_routing.router.compute_query_embedding",
            return_value=[0.4] * 384,
        ), patch("app.services.adaptive_routing.router.MIN_SAMPLES_FOR_DIRECT_ROUTING", 5):
            results = verify_direct_routing({"python": "What is a python tuple?"})

        self.assertEqual(results["python"]["mode"], "full_exploration")
        self.assertEqual(results["python"]["model_count"], 4)


if __name__ == "__main__":
    unittest.main()
