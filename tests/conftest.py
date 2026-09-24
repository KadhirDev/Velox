"""
Shared pytest fixtures for CloudOS-RL test suite.
All fixtures are session- or function-scoped as appropriate.
"""

import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── Project root on path ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.getLogger("shap").setLevel(logging.ERROR)
logging.getLogger("kafka").setLevel(logging.ERROR)
logging.getLogger("confluent_kafka").setLevel(logging.ERROR)


# =============================================================================
# Config fixtures
# =============================================================================

@pytest.fixture(scope="session")
def base_config() -> Dict:
    """Minimal config dict compatible with all modules."""
    return {
        "aws":  {"region": "us-east-1"},
        "kafka": {
            "bootstrap_servers": "localhost:9092",
            "group_id":          "cloudos-test",
            "topics": {
                "decisions": "velox.scheduling.decisions",
                "metrics":   "velox.metrics",
                "alerts":    "velox.alerts",
                "workload":  "velox.workload.events",
            },
        },
        "prometheus": {"host": "0.0.0.0", "port": 9090},
        "bridge": {
            "poll_timeout_seconds":           1.0,
            "max_messages_per_poll":          100,
            "pipeline_metrics_push_interval": 30,
            "decision_window_seconds":        60,
        },
        "data_pipeline": {
            "pricing_refresh_sec":       3600,
            "carbon_refresh_sec":        900,
            "cur_refresh_sec":           3600,
            "pricing_output_path":       "data/pricing/aws_pricing.json",
            "actual_costs_output_path":  "data/pricing/aws_actual_costs.json",
            "carbon_output_path":        "data/carbon/carbon_intensity.json",
            "anomaly_threshold_pct":     50.0,
        },
        "environment_config": {
            "max_episode_steps": 100,
            "n_envs":            1,
        },
        "model": {
            "path":    "models/best/best_model",
            "vecnorm": "models/vec_normalize.pkl",
        },
    }


@pytest.fixture(scope="session")
def tmp_data_dir(tmp_path_factory) -> Path:
    """Session-scoped temp directory with realistic pipeline data files."""
    base = tmp_path_factory.mktemp("velox_data")
    (base / "pricing").mkdir()
    (base / "carbon").mkdir()
    (base / "shap").mkdir()

    # Write minimal but valid pricing JSON (matches Module G output format)
    pricing = {
        "us-east-1":     {"on_demand_per_vcpu_hr": 0.096, "spot_discount": 0.65},
        "eu-north-1":    {"on_demand_per_vcpu_hr": 0.098, "spot_discount": 0.70},
        "eu-west-1":     {"on_demand_per_vcpu_hr": 0.107, "spot_discount": 0.63},
        "us-west-2":     {"on_demand_per_vcpu_hr": 0.096, "spot_discount": 0.68},
        "ap-southeast-1":{"on_demand_per_vcpu_hr": 0.114, "spot_discount": 0.58},
    }
    (base / "pricing" / "aws_pricing.json").write_text(json.dumps(pricing, indent=2))

    # Write minimal but valid carbon JSON (matches Module G output format)
    carbon = {
        "us-east-1":     {"gco2_per_kwh": 415.0, "source": "static"},
        "eu-north-1":    {"gco2_per_kwh":  42.0, "source": "static"},
        "eu-west-1":     {"gco2_per_kwh": 316.0, "source": "static"},
        "us-west-2":     {"gco2_per_kwh": 192.0, "source": "static"},
        "ap-southeast-1":{"gco2_per_kwh": 453.0, "source": "static"},
    }
    (base / "carbon" / "carbon_intensity.json").write_text(json.dumps(carbon, indent=2))

    return base


@pytest.fixture
def config_with_data(base_config, tmp_data_dir) -> Dict:
    """Config that points data_pipeline paths to the temp data directory."""
    cfg = dict(base_config)
    cfg["data_pipeline"] = {
        **base_config["data_pipeline"],
        "pricing_output_path":      str(tmp_data_dir / "pricing" / "aws_pricing.json"),
        "actual_costs_output_path": str(tmp_data_dir / "pricing" / "aws_actual_costs.json"),
        "carbon_output_path":       str(tmp_data_dir / "carbon" / "carbon_intensity.json"),
    }
    return cfg


# =============================================================================
# Workload fixtures
# =============================================================================

@pytest.fixture
def sample_workload() -> Dict:
    """A realistic workload dict for SchedulerAgent.decide() and StateBuilder.build()."""
    return {
        "workload_id":             "test-wl-001",
        "cpu_request_vcpu":        4.0,
        "memory_request_gb":       8.0,
        "gpu_count":               0,
        "storage_gb":              100.0,
        "network_bandwidth_gbps":  1.0,
        "expected_duration_hours": 2.0,
        "priority":                2,
        "sla_latency_ms":          200,
        "sla_tier":                "standard",
        "workload_type":           "training",
        "workload_type_encoded":   0,
        "is_spot_tolerant":        1,
        "constraints":             {},
    }


@pytest.fixture
def sample_pricing() -> Dict:
    return {
        "us-east-1":      0.096,
        "eu-north-1":     0.098,
        "eu-west-1":      0.107,
        "us-west-2":      0.096,
        "eu-central-1":   0.111,
        "ap-southeast-1": 0.114,
        "ap-northeast-1": 0.118,
        "us-central1":    0.096,
        "europe-west4":   0.107,
        "eastus":         0.096,
    }


@pytest.fixture
def sample_carbon() -> Dict:
    return {
        "us-east-1":      415.0,
        "eu-north-1":      42.0,
        "eu-west-1":      316.0,
        "us-west-2":      192.0,
        "eu-central-1":   338.0,
        "ap-southeast-1": 453.0,
        "ap-northeast-1": 506.0,
        "us-central1":    360.0,
        "europe-west4":   284.0,
        "eastus":         400.0,
    }


@pytest.fixture
def sample_state(sample_pricing, sample_carbon) -> np.ndarray:
    """A pre-built (45,) float32 state vector for explainability tests."""
    rng   = np.random.default_rng(42)
    state = rng.uniform(0.0, 1.0, 45).astype(np.float32)
    return state


# =============================================================================
# Mock model fixture
# =============================================================================

@pytest.fixture
def mock_ppo_model():
    """Mock SB3 PPO model with predict + predict_values."""
    import torch
    mock_policy = MagicMock()
    mock_v      = MagicMock()
    mock_v.item.return_value = 2.5
    mock_policy.predict_values.return_value = mock_v

    mock_model = MagicMock()
    mock_model.policy = mock_policy
    mock_model.predict.return_value = (np.array([0, 2, 3, 1, 1, 2]), None)
    return mock_model


# =============================================================================
# CloudWorkload CR fixture
# =============================================================================

@pytest.fixture
def sample_cr() -> Dict:
    return {
        "apiVersion": "cloudos.ai/v1alpha1",
        "kind": "CloudWorkload",
        "metadata": {
            "name":            "fixture-job",
            "namespace":       "cloudos-rl",
            "resourceVersion": "9999",
        },
        "spec": {
            "workloadType":           "training",
            "priority":               2,
            "spotTolerant":           True,
            "expectedDurationHours":  4.0,
            "resources": {
                "cpu":    "8",
                "memory": "32Gi",
                "gpu":    1,
                "storage": "200Gi",
                "networkBandwidthGbps": 2.0,
            },
            "sla": {"maxLatencyMs": 500, "tier": "standard"},
            "constraints": {},
        },
        "status": {"phase": "Pending"},
    }
