"""
Integration / smoke tests for CloudOS-RL.
Tests the full decision pipeline end-to-end with mocked external dependencies.

These tests verify that the MODULES WORK TOGETHER, not just in isolation:
  Module G (data) → Module A (state + SHAP) → Module C (operator) → Module D (Kafka)

No live Kafka, no live AWS, no trained model required.
All external calls are mocked at the boundary.

Run with:
  pytest tests/test_integration.py -v -m integration
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Shared test data ──────────────────────────────────────────────────────────
_PRICING = {
    "us-east-1":      {"on_demand_per_vcpu_hr": 0.096, "spot_discount": 0.65},
    "eu-north-1":     {"on_demand_per_vcpu_hr": 0.098, "spot_discount": 0.70},
    "eu-west-1":      {"on_demand_per_vcpu_hr": 0.107, "spot_discount": 0.63},
    "us-west-2":      {"on_demand_per_vcpu_hr": 0.096, "spot_discount": 0.68},
    "ap-southeast-1": {"on_demand_per_vcpu_hr": 0.114, "spot_discount": 0.58},
}
_CARBON = {
    "us-east-1":      {"gco2_per_kwh": 415.0},
    "eu-north-1":     {"gco2_per_kwh":  42.0},
    "eu-west-1":      {"gco2_per_kwh": 316.0},
    "us-west-2":      {"gco2_per_kwh": 192.0},
    "ap-southeast-1": {"gco2_per_kwh": 453.0},
}


def _make_data_dir():
    """Creates a temp directory with valid pipeline output files."""
    tmpdir = Path(tempfile.mkdtemp())
    (tmpdir / "pricing").mkdir()
    (tmpdir / "carbon").mkdir()
    (tmpdir / "pricing" / "aws_pricing.json").write_text(json.dumps(_PRICING))
    (tmpdir / "carbon" / "carbon_intensity.json").write_text(json.dumps(_CARBON))
    return tmpdir


def _make_config(tmpdir: Path) -> dict:
    return {
        "aws":    {"region": "us-east-1"},
        "kafka":  {
            "bootstrap_servers": "localhost:9092",
            "group_id": "cloudos-test",
            "topics": {
                "decisions": "velox.scheduling.decisions",
                "metrics":   "velox.metrics",
                "alerts":    "velox.alerts",
                "workload":  "velox.workload.events",
            },
        },
        "data_pipeline": {
            "pricing_output_path":      str(tmpdir / "pricing" / "aws_pricing.json"),
            "actual_costs_output_path": str(tmpdir / "pricing" / "aws_actual_costs.json"),
            "carbon_output_path":       str(tmpdir / "carbon" / "carbon_intensity.json"),
        },
        "environment_config": {"max_episode_steps": 100},
        "model": {"path": "models/best/best_model", "vecnorm": "models/vec_normalize.pkl"},
    }


@pytest.mark.integration
class TestDataToStateIntegration(unittest.TestCase):
    """Module G output files → StateBuilder → valid 45-dim state."""

    def setUp(self):
        self.tmpdir = _make_data_dir()
        self.config = _make_config(self.tmpdir)

    def test_pricing_file_to_state(self):
        """Pricing file → PricingCache → StateBuilder → (45,) state."""
        from ai_engine.cloud_adapter.pricing_cache import PricingCache
        from ai_engine.environment.state_builder   import StateBuilder

        cache    = PricingCache(self.config)
        builder  = StateBuilder(self.config)
        pricing  = cache.get_current_pricing()

        carbon = {r: e["gco2_per_kwh"] for r, e in _CARBON.items()}

        workload = {
            "cpu_request_vcpu": 4.0, "memory_request_gb": 8.0,
            "gpu_count": 0, "storage_gb": 100.0,
            "network_bandwidth_gbps": 1.0, "expected_duration_hours": 2.0,
            "priority": 2, "sla_latency_ms": 200,
            "workload_type_encoded": 0, "is_spot_tolerant": 1,
        }
        state = builder.build(workload, pricing, carbon, [])
        self.assertEqual(state.shape, (45,))
        self.assertFalse(np.any(np.isnan(state)))
        self.assertFalse(np.any(np.isinf(state)))


@pytest.mark.integration
class TestOperatorToKafkaIntegration(unittest.TestCase):
    """Operator heuristic decision → Kafka producer (mocked)."""

    def setUp(self):
        self.tmpdir = _make_data_dir()
        self.config = _make_config(self.tmpdir)

    def test_operator_publishes_to_kafka(self):
        """Operator heuristic path must call publish_decision() for a spot-tolerant workload."""
        from ai_engine.operator.operator import VeloxOperator

        op = VeloxOperator(
            config=self.config,
            dry_run=False,
            no_kafka=False,
            no_shap=True,
        )

        # Inject a mock producer directly — bypasses _resolve_bootstrap entirely
        mock_producer = MagicMock()
        publish_calls = []

        def fake_publish_decision(decision):
            publish_calls.append(decision)
            return True

        mock_producer.publish_decision = fake_publish_decision
        op._producer = mock_producer

        cr = {
            "metadata": {"name": "kafka-test", "namespace": "cloudos-rl", "resourceVersion": "1"},
            "spec": {
                "workloadType": "batch", "priority": 1,
                "spotTolerant": True, "expectedDurationHours": 1.0,
                "resources": {"cpu": "2", "memory": "4Gi", "gpu": 0,
                              "storage": "50Gi", "networkBandwidthGbps": 1.0},
                "sla": {"maxLatencyMs": 500, "tier": "best_effort"},
                "constraints": {},
            },
            "status": {"phase": "Pending"},
        }
        op._list_pending = MagicMock(return_value=[cr])

        # Patch _load_agent so run_once() does not load the real PPO model.
        # Keeping _agent=None forces _make_decision() -> _heuristic_decision().
        with patch.object(op, "_load_agent", return_value=None):
            with patch("subprocess.run") as mock_sub:
                mock_sub.return_value = MagicMock(returncode=0, stdout="patched", stderr="")
                op.run_once()

        self.assertGreater(len(publish_calls), 0, "publish_decision() was never called")
        decision = publish_calls[0]
        self.assertEqual(
            decision["region"], "eu-north-1",
            f"Heuristic should pick eu-north-1 for spot-tolerant workload, got {decision['region']}",
        )


@pytest.mark.integration
class TestBackgroundGenToSHAPIntegration(unittest.TestCase):
    """BackgroundDataGenerator → SHAPExplainer → ExplanationFormatter."""

    def test_background_has_16_region_coverage(self):
        """Background dataset should include samples covering multiple regions."""
        from ai_engine.explainability.background_generator import BackgroundDataGenerator
        config = {
            "data_pipeline": {
                "pricing_output_path": "data/pricing/aws_pricing.json",
                "carbon_output_path":  "data/carbon/carbon_intensity.json",
            }
        }
        gen = BackgroundDataGenerator(config)
        bg  = gen.generate(n_samples=50, seed=42, force=True)
        self.assertEqual(bg.shape, (50, 45))
        # Pricing dims [10:20] should not all be identical (region diversity)
        pricing_dims = bg[:, 10:20]
        self.assertGreater(pricing_dims.std(), 0,
                           "All pricing dims identical — no region diversity in background")

    def test_formatter_handles_zero_shap_values(self):
        """Formatter must not crash when all SHAP values are zero."""
        from ai_engine.explainability.explanation_formatter import ExplanationFormatter
        formatter = ExplanationFormatter()
        zero_shap = {
            "top_drivers":    [],
            "base_value":     0.0,
            "shap_values":    {f"feat_{i}": 0.0 for i in range(45)},
            "top_positive":   [],
            "top_negative":   [],
            "explanation_ms": 0.0,
            "state_mean":     0.0,
            "state_std":      0.0,
        }
        decision = {"cloud": "aws", "region": "us-east-1",
                    "purchase_option": "on_demand", "instance_type": "m5.large"}
        result = formatter.format(zero_shap, decision)
        self.assertIn("summary", result)
        self.assertIsInstance(result["confidence"], float)


@pytest.mark.integration
class TestWorkloadMapperToHeuristicIntegration(unittest.TestCase):
    """WorkloadMapper → VeloxOperator._heuristic_decision correctness."""

    def test_spot_tolerant_training_job_goes_to_clean_region(self):
        from ai_engine.operator.workload_mapper import WorkloadMapper
        from ai_engine.operator.operator        import VeloxOperator

        config = {"data_pipeline": {
            "pricing_output_path": "data/pricing/aws_pricing.json",
            "carbon_output_path":  "data/carbon/carbon_intensity.json",
        }}
        mapper  = WorkloadMapper()
        op      = VeloxOperator(config, dry_run=True, no_kafka=True, no_shap=True)
        op._agent = None

        cr = {
            "metadata": {"name": "green-job", "namespace": "cloudos-rl", "resourceVersion": "1"},
            "spec": {
                "workloadType": "training", "priority": 2,
                "spotTolerant": True, "expectedDurationHours": 4.0,
                "resources": {"cpu": "8", "memory": "32Gi", "gpu": 1,
                              "storage": "200Gi", "networkBandwidthGbps": 2.0},
                "sla": {"maxLatencyMs": 500, "tier": "standard"},
                "constraints": {"maxCarbonGco2PerKwh": 200.0},
            },
            "status": {"phase": "Pending"},
        }

        workload = mapper.map(cr)
        self.assertIsNotNone(workload)
        self.assertEqual(workload["is_spot_tolerant"], 1)

        decision = op._heuristic_decision(workload)
        self.assertEqual(decision["region"],         "eu-north-1")
        self.assertEqual(decision["purchase_option"],"spot")
        self.assertGreater(decision["carbon_savings_pct"], 0)

    def test_latency_critical_inference_stays_on_demand(self):
        from ai_engine.operator.workload_mapper import WorkloadMapper
        from ai_engine.operator.operator        import VeloxOperator

        config = {"data_pipeline": {
            "pricing_output_path": "data/pricing/aws_pricing.json",
            "carbon_output_path":  "data/carbon/carbon_intensity.json",
        }}
        mapper = WorkloadMapper()
        op     = VeloxOperator(config, dry_run=True, no_kafka=True, no_shap=True)
        op._agent = None

        cr = {
            "metadata": {"name": "latency-job", "namespace": "cloudos-rl", "resourceVersion": "2"},
            "spec": {
                "workloadType": "inference", "priority": 4,
                "spotTolerant": False, "expectedDurationHours": 720.0,
                "resources": {"cpu": "4", "memory": "8Gi", "gpu": 0,
                              "storage": "50Gi", "networkBandwidthGbps": 5.0},
                "sla": {"maxLatencyMs": 50, "tier": "critical"},
                "constraints": {},
            },
            "status": {"phase": "Pending"},
        }

        workload = mapper.map(cr)
        decision = op._heuristic_decision(workload)
        self.assertEqual(decision["purchase_option"], "on_demand")


@pytest.mark.integration
class TestStatusPatchSchema(unittest.TestCase):
    """Verifies the status patch body matches the CRD status schema."""

    def test_scheduled_status_has_all_crd_fields(self):
        """set_scheduled must produce all fields declared in crd.yaml status."""
        from ai_engine.operator.status_writer import StatusWriter
        captured = {}

        writer = StatusWriter(dry_run=False)
        with patch("subprocess.run") as mock_sub:
            def capture_run(cmd, **kw):
                patch_str = cmd[cmd.index("-p") + 1]
                captured["patch"] = json.loads(patch_str)
                return MagicMock(returncode=0, stdout="", stderr="")
            mock_sub.side_effect = capture_run

            writer.set_scheduled("test-job", "cloudos-rl", {
                "cloud":               "aws",
                "region":              "eu-north-1",
                "instance_type":       "m5.large",
                "purchase_option":     "spot",
                "sla_tier":            "standard",
                "estimated_cost_per_hr": 0.032,
                "cost_savings_pct":    66.7,
                "carbon_savings_pct":  89.9,
                "latency_ms":          45.2,
                "decision_id":         "uuid-test",
                "explanation": {
                    "summary":        "Test",
                    "top_drivers":    [],
                    "confidence":     0.9,
                    "explanation_ms": 85.0,
                },
            })

        status = captured.get("patch", {}).get("status", {})
        crd_fields = [
            "phase", "scheduledCloud", "scheduledRegion",
            "instanceType", "purchaseOption", "estimatedCostPerHr",
            "costSavingsPct", "carbonSavingsPct", "schedulingLatencyMs",
            "decisionId", "scheduledAt", "message",
        ]
        for field in crd_fields:
            self.assertIn(field, status, f"CRD status field '{field}' missing from patch")

    def test_cost_savings_pct_is_percent_string(self):
        """costSavingsPct must be formatted as '66.7%' not 0.667."""
        from ai_engine.operator.status_writer import StatusWriter
        captured = {}
        with patch("subprocess.run") as mock_sub:
            def capture_run(cmd, **kw):
                patch_str = cmd[cmd.index("-p") + 1]
                captured["patch"] = json.loads(patch_str)
                return MagicMock(returncode=0, stdout="", stderr="")
            mock_sub.side_effect = capture_run
            StatusWriter(dry_run=False).set_scheduled("j", "ns", {
                "cloud": "aws", "region": "eu-north-1",
                "instance_type": "m5.large", "purchase_option": "spot",
                "sla_tier": "standard", "estimated_cost_per_hr": 0.032,
                "cost_savings_pct": 66.7, "carbon_savings_pct": 89.9,
                "latency_ms": 1.5, "decision_id": "x", "explanation": {},
            })
        val = captured.get("patch", {}).get("status", {}).get("costSavingsPct", "")
        self.assertIn("%", str(val), f"costSavingsPct should be '66.7%', got '{val}'")


if __name__ == "__main__":
    unittest.main(verbosity=2)
