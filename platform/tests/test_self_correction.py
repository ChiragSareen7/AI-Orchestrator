"""Tests for requirement-driven self-correction loop."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.services.adaptive_routing import storage as routing_storage
from app.services.self_correction import config as sc_config
from app.services.self_correction import logger as sc_logger
from app.services.self_correction import requirements as req_module
from app.services.self_correction.loop import run_self_correction_loop
from app.services.self_correction.prompt_enhancer import enhance_prompt


class SelfCorrectionTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        store = Path(self._tmpdir)

        routing_storage.STORE = store
        routing_storage.ROUTING_TABLE_PATH = store / "routing_table.json"
        routing_storage.ROUTING_LOGS_PATH = store / "routing_logs.json"
        routing_storage.ensure_routing_files()

        req_module.STORE = store
        req_module.PROFILES_PATH = store / "requirements_profiles.json"
        req_module.ensure_profiles_file()

        sc_logger.STORE = store
        sc_logger.SELF_CORRECTION_LOGS_PATH = store / "self_correction_logs.json"
        sc_logger.ensure_self_correction_logs()

        self.profile = req_module.save_requirements_profile(
            "test_client",
            {
                "tone_style": "concise and professional",
                "must_include": "examples",
                "must_avoid": "speculation",
            },
        )

        self._call_counts: dict[str, int] = {}

    def tearDown(self) -> None:
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _fake_execute(self, model_name: str, original_query: str, prompt_text: str) -> dict:
        self._call_counts[model_name] = self._call_counts.get(model_name, 0) + 1
        return {
            "model": model_name,
            "query": original_query,
            "response": f"Answer from {model_name}: {prompt_text[:40]}",
            "error": None,
            "latency": 50.0,
            "tokenUsage": 10,
        }

    def _make_judge_fn(self, outcomes: list[dict]):
        calls = {"n": 0}

        def judge(query, response, profile):
            idx = min(calls["n"], len(outcomes) - 1)
            calls["n"] += 1
            return outcomes[idx]

        return judge

    def _fake_evaluate(self, *args, **kwargs):
        response = args[1] if len(args) > 1 else kwargs.get("response", "")
        score = 0.9 if "python_model" in response else 0.5
        return {
            "accuracyScore": score,
            "relevanceScore": score,
            "hallucinationScore": 0.1,
            "confidenceScore": score,
            "latency": 50.0,
            "tokenUsage": 10,
            "cost": 0.0,
            "toxicityScore": 0.0,
            "errorRate": 0.0,
        }

    def _run(self, judge_outcomes: list[dict], query: str = "Explain Python lists"):
        with patch(
            "app.services.self_correction.loop.plan_query_execution",
            return_value={
                "cluster_id": "python",
                "routing_decision": {"cluster_id": "python", "mode": "direct"},
            },
        ), patch(
            "app.services.self_correction.loop.get_ranked_models_for_cluster",
            side_effect=lambda cluster_id, exclude=None: [
                m
                for m in ["groq_model", "organic_model", "gita_model", "python_model"]
                if not exclude or m not in exclude
            ],
        ), patch(
            "app.services.self_correction.loop.evaluate_response",
            side_effect=self._fake_evaluate,
        ):
            return run_self_correction_loop(
                query=query,
                client_id="test_client",
                judge_fn=self._make_judge_fn(judge_outcomes),
                execute_fn=self._fake_execute,
            )

    def test_pass_on_attempt_1_stops_immediately(self) -> None:
        result = self._run([{"pass": True, "score": 0.95, "reasons": []}])
        self.assertTrue(result["requirements_met"])
        self.assertEqual(result["passed_on_attempt"], 1)
        self.assertFalse(result["requirements_not_met"])
        self.assertEqual(len(result["attempts"]), 1)

    def test_attempt_2_same_model_refined_prompt(self) -> None:
        result = self._run(
            [
                {"pass": False, "score": 0.3, "reasons": ["missing examples"]},
                {"pass": True, "score": 0.9, "reasons": []},
            ]
        )
        self.assertEqual(result["passed_on_attempt"], 2)
        self.assertEqual(result["attempts"][0]["model"], result["attempts"][1]["model"])
        self.assertNotEqual(result["attempts"][1]["prompt"], result["attempts"][0]["prompt"])
        self.assertIn("missing examples", result["attempts"][1]["prompt"])

    def test_attempt_3_switches_model_with_original_query(self) -> None:
        result = self._run(
            [
                {"pass": False, "score": 0.2, "reasons": ["tone mismatch"]},
                {"pass": False, "score": 0.3, "reasons": ["still wrong tone"]},
                {"pass": True, "score": 0.88, "reasons": []},
            ]
        )
        self.assertEqual(result["passed_on_attempt"], 3)
        self.assertNotEqual(result["attempts"][2]["model"], result["attempts"][0]["model"])
        self.assertEqual(result["attempts"][2]["prompt"], "Explain Python lists")

    def test_attempt_4_refines_new_model_prompt(self) -> None:
        result = self._run(
            [
                {"pass": False, "score": 0.2, "reasons": ["a"]},
                {"pass": False, "score": 0.3, "reasons": ["b"]},
                {"pass": False, "score": 0.4, "reasons": ["missing format"]},
                {"pass": True, "score": 0.91, "reasons": []},
            ]
        )
        self.assertEqual(result["passed_on_attempt"], 4)
        self.assertEqual(result["attempts"][3]["model"], result["attempts"][2]["model"])
        self.assertIn("missing format", result["attempts"][3]["prompt"])

    def test_all_four_fail_returns_best_with_flag(self) -> None:
        result = self._run(
            [
                {"pass": False, "score": 0.2, "reasons": ["r1"]},
                {"pass": False, "score": 0.5, "reasons": ["r2"]},
                {"pass": False, "score": 0.4, "reasons": ["r3"]},
                {"pass": False, "score": 0.3, "reasons": ["r4"]},
            ]
        )
        self.assertIsNone(result["passed_on_attempt"])
        self.assertTrue(result["requirements_not_met"])
        self.assertIn("best available", result["message"].lower())
        self.assertEqual(len(result["attempts"]), 4)
        self.assertEqual(result["final_judge"]["score"], 0.5)

    def test_requirements_profile_persists_and_reloads(self) -> None:
        loaded = req_module.get_requirements_profile("test_client")
        assert loaded is not None
        self.assertEqual(loaded["tone_style"], "concise and professional")

        updated = req_module.save_requirements_profile(
            "test_client", {**loaded, "detail_level": "high"}
        )
        self.assertEqual(updated["detail_level"], "high")

        raw = json.loads(req_module.PROFILES_PATH.read_text())
        self.assertEqual(raw["test_client"]["detail_level"], "high")

    def test_enhance_prompt_uses_specific_reasons(self) -> None:
        prompt = enhance_prompt(
            "What is Python?",
            self.profile,
            ["missing examples", "tone too casual"],
        )
        self.assertIn("missing examples", prompt)
        self.assertIn("tone too casual", prompt)
        self.assertNotIn("be more accurate", prompt.lower())

    def test_attempt_1_calls_all_four_models(self) -> None:
        self._run([{"pass": True, "score": 1.0, "reasons": []}])
        for model in ["organic_model", "python_model", "gita_model", "groq_model"]:
            self.assertGreaterEqual(self._call_counts.get(model, 0), 1)

    def test_logs_written_to_json(self) -> None:
        self._run([{"pass": True, "score": 0.9, "reasons": []}])
        logs = sc_logger.read_self_correction_logs()
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]["client_id"], "test_client")
        self.assertIn("attempts", logs[0])

    def test_missing_profile_rejected_when_required(self) -> None:
        with patch("app.services.self_correction.loop.REQUIREMENTS_PROFILE_REQUIRED", True):
            with self.assertRaises(ValueError):
                run_self_correction_loop(
                    query="test",
                    client_id="nonexistent_client",
                    judge_fn=lambda q, r, p: {"pass": True, "score": 1.0, "reasons": []},
                    execute_fn=self._fake_execute,
                )


if __name__ == "__main__":
    unittest.main()
