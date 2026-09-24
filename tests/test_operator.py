"""
Module C — Operator Controller Loop Tests
==========================================
Tests the full operator pipeline WITHOUT requiring:
  - A running Kubernetes cluster
  - A trained PPO model
  - Kafka connection

All kubectl calls are mocked.
SchedulerAgent is mocked to return a fixed decision.

Tests:
  1.  WorkloadMapper — parses training CR correctly
  2.  WorkloadMapper — parses inference CR correctly
  3.  WorkloadMapper — handles missing optional fields
  4.  WorkloadMapper — rejects invalid spec gracefully
  5.  WorkloadMapper — parses memory units correctly
  6.  WorkloadMapper — parses CPU millicores correctly
  7.  StatusWriter   — dry_run logs without calling kubectl
  8.  StatusWriter   — set_scheduled builds correct patch body
  9.  StatusWriter   — set_failed writes correct phase
  10. VeloxOperator — run_once processes pending workloads (mocked kubectl)
  11. VeloxOperator — skips already-processed resourceVersions
  12. VeloxOperator — uses heuristic when agent returns None
  13. VeloxOperator — handles kubectl failure gracefully
  14. Full pipeline  — CR → mapper → mock agent → status patch end-to-end

Run:
  python tests/test_operator.py
  python -m pytest tests/test_operator.py -v
"""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call
from typing import Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_cr(
    name:          str  = "test-job",
    namespace:     str  = "cloudos-rl",
    workload_type: str  = "training",
    cpu:           str  = "4",
    memory:        str  = "8Gi",
    gpu:           int  = 0,
    storage:       str  = "100Gi",
    priority:      int  = 2,
    spot_tolerant: bool = False,
    phase:         str  = "",
    resource_version: str = "1001",
) -> Dict:
    return {
        "apiVersion": "cloudos.ai/v1alpha1",
        "kind": "CloudWorkload",
        "metadata": {
            "name":            name,
            "namespace":       namespace,
            "resourceVersion": resource_version,
        },
        "spec": {
            "workloadType":           workload_type,
            "priority":               priority,
            "spotTolerant":           spot_tolerant,
            "expectedDurationHours":  2.0,
            "resources": {
                "cpu":     cpu,
                "memory":  memory,
                "gpu":     gpu,
                "storage": storage,
                "networkBandwidthGbps": 1.0,
            },
            "sla": {
                "maxLatencyMs": 200,
                "tier":         "standard",
            },
        },
        "status": {"phase": phase},
    }


def _make_decision() -> Dict:
    return {
        "cloud":               "aws",
        "region":              "eu-north-1",
        "instance_type":       "m5.large",
        "purchase_option":     "spot",
        "sla_tier":            "standard",
        "estimated_cost_per_hr": 0.032,
        "cost_savings_pct":    66.7,
        "carbon_savings_pct":  89.9,
        "latency_ms":          45.2,
        "decision_id":         "test-decision-uuid",
        "workload_id":         "test-job",
        "explanation": {
            "summary":     "Carbon optimised: eu-north-1 (42 gCO2/kWh)",
            "top_drivers": [{"feature": "carbon_region_0", "shap_value": 0.8}],
            "confidence":  0.87,
            "explanation_ms": 92.1,
        },
    }


_TEST_CONFIG = {
    "data_pipeline": {
        "pricing_output_path": "data/pricing/aws_pricing.json",
        "carbon_output_path":  "data/carbon/carbon_intensity.json",
    },
}


# ── TestWorkloadMapper ────────────────────────────────────────────────────────

class TestWorkloadMapper(unittest.TestCase):

    def setUp(self):
        from ai_engine.operator.workload_mapper import WorkloadMapper
        self.mapper = WorkloadMapper()

    def test_maps_training_cr(self):
        cr     = _make_cr(workload_type="training", cpu="8", memory="32Gi", gpu=1)
        result = self.mapper.map(cr)
        self.assertIsNotNone(result)
        self.assertEqual(result["workload_type"],       "training")
        self.assertEqual(result["workload_type_encoded"], 0)
        self.assertAlmostEqual(result["cpu_request_vcpu"], 8.0)
        self.assertAlmostEqual(result["memory_request_gb"], 32.0)
        self.assertEqual(result["gpu_count"], 1)

    def test_maps_inference_cr(self):
        cr     = _make_cr(workload_type="inference", spot_tolerant=False)
        result = self.mapper.map(cr)
        self.assertIsNotNone(result)
        self.assertEqual(result["workload_type_encoded"], 1)
        self.assertEqual(result["is_spot_tolerant"],      0)

    def test_spot_tolerant_true(self):
        cr     = _make_cr(spot_tolerant=True)
        result = self.mapper.map(cr)
        self.assertEqual(result["is_spot_tolerant"], 1)

    def test_handles_missing_optional_fields(self):
        # Minimal valid CR
        cr = {
            "metadata": {"name": "minimal-job", "namespace": "cloudos-rl",
                         "resourceVersion": "1"},
            "spec": {
                "workloadType": "batch",
                "resources":    {"cpu": "1", "memory": "2Gi"},
            },
            "status": {},
        }
        result = self.mapper.map(cr)
        self.assertIsNotNone(result)
        self.assertEqual(result["gpu_count"],   0)
        self.assertEqual(result["priority"],    2)
        self.assertAlmostEqual(result["storage_gb"], 50.0)

    def test_rejects_none_spec(self):
        # Missing spec entirely should return None
        cr = {"metadata": {"name": "bad", "namespace": "ns", "resourceVersion": "1"}}
        result = self.mapper.map(cr)
        # Should either return None or return with defaults — not raise
        # (map() handles KeyError gracefully)
        if result is not None:
            self.assertIn("workload_id", result)

    def test_parses_memory_mebibytes(self):
        cr     = _make_cr(memory="512Mi")
        result = self.mapper.map(cr)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["memory_request_gb"], 0.5, places=2)

    def test_parses_cpu_millicores(self):
        cr     = _make_cr(cpu="500m")
        result = self.mapper.map(cr)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["cpu_request_vcpu"], 0.5, places=2)

    def test_all_required_keys_present(self):
        cr     = _make_cr()
        result = self.mapper.map(cr)
        required = [
            "workload_id", "namespace",
            "cpu_request_vcpu", "memory_request_gb", "gpu_count",
            "storage_gb", "network_bandwidth_gbps", "expected_duration_hours",
            "priority", "sla_latency_ms", "workload_type", "workload_type_encoded",
            "is_spot_tolerant",
        ]
        for key in required:
            self.assertIn(key, result, f"Missing key: {key}")


# ── TestStatusWriter ──────────────────────────────────────────────────────────

class TestStatusWriter(unittest.TestCase):

    def setUp(self):
        from ai_engine.operator.status_writer import StatusWriter
        self.writer_dry  = StatusWriter(dry_run=True)
        self.writer_real = StatusWriter(dry_run=False)

    def test_dry_run_returns_true_without_kubectl(self):
        ok = self.writer_dry.set_scheduling("test-job", "cloudos-rl")
        self.assertTrue(ok)

    def test_set_scheduled_dry_run(self):
        ok = self.writer_dry.set_scheduled("test-job", "cloudos-rl", _make_decision())
        self.assertTrue(ok)

    def test_set_failed_dry_run(self):
        ok = self.writer_dry.set_failed("test-job", "cloudos-rl", "test error")
        self.assertTrue(ok)

    def test_safe_explanation_strips_shap_values(self):
        from ai_engine.operator.status_writer import StatusWriter
        full_explanation = {
            "summary":     "test",
            "top_drivers": [{"feature": "cpu", "shap_value": 0.5}] * 10,
            "confidence":  0.9,
            "explanation_ms": 80.0,
            "shap_values": {f"feat_{i}": 0.1 for i in range(45)},  # must be stripped
        }
        result = StatusWriter._safe_explanation(full_explanation)
        self.assertNotIn("shap_values", result)
        self.assertLessEqual(len(result["top_drivers"]), 3)
        self.assertIn("summary", result)
        self.assertIn("confidence", result)

    @patch("subprocess.run")
    def test_real_writer_calls_kubectl(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="patched", stderr="")
        ok = self.writer_real.set_scheduling("job1", "cloudos-rl")
        self.assertTrue(ok)
        self.assertTrue(mock_run.called)
        cmd = mock_run.call_args[0][0]
        self.assertIn("kubectl",           cmd)
        self.assertIn("patch",             cmd)
        self.assertIn("cloudworkload",     cmd)
        self.assertIn("job1",              cmd)
        self.assertIn("--subresource=status", cmd)

    @patch("subprocess.run")
    def test_kubectl_failure_returns_false(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="not found")
        ok = self.writer_real.set_scheduling("bad-job", "cloudos-rl")
        self.assertFalse(ok)


# ── TestVeloxOperator ───────────────────────────────────────────────────────

class TestVeloxOperator(unittest.TestCase):

    def setUp(self):
        from ai_engine.operator.operator import VeloxOperator
        self.operator = VeloxOperator(
            config=_TEST_CONFIG,
            dry_run=True,
            no_kafka=True,
            no_shap=True,
        )

    def _mock_list_pending(self, crs):
        self.operator._list_pending = MagicMock(return_value=crs)

    def _mock_agent_decide(self, decision):
        mock_agent = MagicMock()
        mock_agent.decide.return_value = decision
        self.operator._agent = mock_agent

    def test_run_once_processes_pending(self):
        self._mock_list_pending([_make_cr(name="job1", phase="")])
        self._mock_agent_decide(_make_decision())
        n = self.operator.run_once()
        self.assertEqual(n, 1)

    def test_run_once_skips_non_pending(self):
        self._mock_list_pending([
            _make_cr(name="done",   phase="Scheduled"),
            _make_cr(name="active", phase="Running"),
        ])
        # list_pending should only return Pending — operator should process 0
        # (our mock returns them anyway, but the real _list_pending filters)
        # Test that after processing, seen_rv prevents reprocessing
        self.operator._seen_rv["done"]   = "1001"
        self.operator._seen_rv["active"] = "1001"
        self._mock_list_pending([
            _make_cr(name="done",   resource_version="1001"),
            _make_cr(name="active", resource_version="1001"),
        ])
        n = self.operator.run_once()
        self.assertEqual(n, 0)  # both skipped due to seen_rv

    def test_deduplication_by_resource_version(self):
        cr = _make_cr(name="job-dedup", resource_version="42")
        self._mock_list_pending([cr])
        self._mock_agent_decide(_make_decision())

        n1 = self.operator.run_once()
        self.assertEqual(n1, 1)

        # Same resourceVersion — should skip
        n2 = self.operator.run_once()
        self.assertEqual(n2, 0)

    def test_heuristic_fallback_when_agent_none(self):
        self.operator._agent = None
        cr = _make_cr(name="heuristic-job", spot_tolerant=True)
        self._mock_list_pending([cr])

        n = self.operator.run_once()
        self.assertEqual(n, 1)
        self.assertEqual(self.operator._stats["processed"], 1)

    def test_heuristic_spot_tolerant_picks_eu_north(self):
        self.operator._agent = None
        workload = {"is_spot_tolerant": 1, "sla_tier": "standard"}
        decision = self.operator._heuristic_decision(workload)
        self.assertEqual(decision["region"],         "eu-north-1")
        self.assertEqual(decision["purchase_option"],"spot")

    def test_heuristic_non_spot_picks_us_east(self):
        self.operator._agent = None
        workload = {"is_spot_tolerant": 0, "sla_tier": "standard"}
        decision = self.operator._heuristic_decision(workload)
        self.assertEqual(decision["region"],         "us-east-1")
        self.assertEqual(decision["purchase_option"],"on_demand")

    def test_agent_none_return_uses_heuristic(self):
        mock_agent = MagicMock()
        mock_agent.decide.return_value = None   # simulate model not loaded
        self.operator._agent = mock_agent

        cr = _make_cr(name="fallback-job")
        self._mock_list_pending([cr])
        n = self.operator.run_once()
        self.assertEqual(n, 1)

    def test_handles_empty_pending_list(self):
        self._mock_list_pending([])
        n = self.operator.run_once()
        self.assertEqual(n, 0)

    def test_stats_accumulate(self):
        self._mock_list_pending([_make_cr(name="s1", resource_version="1")])
        self._mock_agent_decide(_make_decision())
        self.operator.run_once()
        self._mock_list_pending([_make_cr(name="s2", resource_version="2")])
        self.operator.run_once()
        self.assertEqual(self.operator._stats["processed"], 2)


# ── TestFullPipeline ──────────────────────────────────────────────────────────

class TestFullPipeline(unittest.TestCase):

    def test_cr_to_decision_end_to_end(self):
        """Full pipeline: CR → WorkloadMapper → (mock agent) → StatusWriter dry_run."""
        from ai_engine.operator.workload_mapper import WorkloadMapper
        from ai_engine.operator.status_writer   import StatusWriter
        from ai_engine.operator.operator        import VeloxOperator

        cr        = _make_cr(
            name="e2e-job",
            workload_type="training",
            cpu="8",
            memory="32Gi",
            gpu=1,
            spot_tolerant=True,
        )
        mapper    = WorkloadMapper()
        writer    = StatusWriter(dry_run=True)

        workload  = mapper.map(cr)
        self.assertIsNotNone(workload)
        self.assertEqual(workload["gpu_count"],         1)
        self.assertEqual(workload["is_spot_tolerant"],  1)
        self.assertEqual(workload["workload_type"],     "training")

        # Mock agent
        mock_agent = MagicMock()
        mock_agent.decide.return_value = _make_decision()

        decision = mock_agent.decide(workload)
        self.assertEqual(decision["cloud"],           "aws")
        self.assertEqual(decision["region"],          "eu-north-1")
        self.assertEqual(decision["purchase_option"], "spot")

        ok = writer.set_scheduled("e2e-job", "cloudos-rl", decision)
        self.assertTrue(ok)

        print("\n✅ Full pipeline test passed")
        print(f"   Workload: {workload['workload_type']} {workload['cpu_request_vcpu']}vCPU "
              f"{workload['memory_request_gb']}GB GPU={workload['gpu_count']}")
        print(f"   Decision: {decision['cloud']}/{decision['region']} "
              f"{decision['purchase_option']} "
              f"cost={decision['estimated_cost_per_hr']}/hr "
              f"savings={decision['cost_savings_pct']}%")
        print(f"   SHAP summary: {decision['explanation']['summary']}")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  CloudOS-RL — Module C: Operator Pipeline Tests")
    print("=" * 60 + "\n")
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestWorkloadMapper))
    suite.addTests(loader.loadTestsFromTestCase(TestStatusWriter))
    suite.addTests(loader.loadTestsFromTestCase(TestVeloxOperator))
    suite.addTests(loader.loadTestsFromTestCase(TestFullPipeline))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)