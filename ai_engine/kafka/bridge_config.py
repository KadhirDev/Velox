"""
Bridge Configuration Loader
=============================
Loads config for the Kafka → Prometheus bridge from settings.yaml.
Falls back to safe defaults if any section is missing.

Config section in settings.yaml:

  prometheus:
    port: 9090
    host: 0.0.0.0

  kafka:
    bootstrap_servers: localhost:9092
    group_id: velox-consumers

  bridge:
    poll_timeout_seconds: 1.0
    max_messages_per_poll: 100
    pipeline_metrics_push_interval: 30
    decision_window_seconds: 60
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

try:
    import yaml
except ImportError:
    raise ImportError("PyYAML not installed. Run: pip install pyyaml")

logger = logging.getLogger(__name__)

_DEFAULTS: Dict[str, Any] = {
    "kafka": {
        "bootstrap_servers": "localhost:9092",
        "group_id": "velox-consumers",
        "topics": {
            "decisions": "velox.scheduling.decisions",
            "metrics": "velox.metrics",
            "alerts": "velox.alerts",
            "workload": "velox.workload.events",
        },
    },
    "prometheus": {
        "host": "0.0.0.0",
        "port": 9090,
    },
    "bridge": {
        "poll_timeout_seconds": 1.0,
        "max_messages_per_poll": 100,
        "pipeline_metrics_push_interval": 30,
        "decision_window_seconds": 60,
    },
    "data_pipeline": {
        "carbon_output_path": "data/carbon/carbon_intensity.json",
        "pricing_output_path": "data/pricing/aws_pricing.json",
    },
}


class BridgeConfig:
    """
    Typed config accessor for the Kafka-Prometheus bridge.
    All fields have safe defaults so the bridge starts even if settings.yaml
    is incomplete.
    """

    def __init__(self, config: Dict[str, Any]):
        self._raw = config
        self._config = config

        kafka = config.get("kafka", {}) or {}
        bridge = config.get("bridge", {}) or {}
        prom = config.get("prometheus", {}) or {}
        dp = config.get("data_pipeline", {}) or {}

        # Env var override takes highest priority for runtime flexibility
        env_bootstrap = os.environ.get("CLOUDOS_KAFKA_BOOTSTRAP", "").strip()
        config_bootstrap = kafka.get("bootstrap_servers", "localhost:9092")

        # Attributes accessed directly by tests / runtime
        self.bootstrap_servers = env_bootstrap or config_bootstrap
        self.topics = kafka.get(
            "topics",
            {
                "decisions": "velox.scheduling.decisions",
                "metrics": "velox.metrics",
                "alerts": "velox.alerts",
                "workload": "velox.workload.events",
            },
        )
        self.poll_timeout_seconds = float(bridge.get("poll_timeout_seconds", 1.0))
        self.max_messages_per_poll = int(bridge.get("max_messages_per_poll", 100))
        self.pipeline_metrics_push_interval = int(
            bridge.get("pipeline_metrics_push_interval", 30)
        )
        self.decision_window_seconds = int(
            bridge.get("decision_window_seconds", 60)
        )
        self.prometheus_host = prom.get("host", "0.0.0.0")
        self.prometheus_port = int(prom.get("port", 9090))

        self.pricing_path = dp.get(
            "pricing_output_path", "data/pricing/aws_pricing.json"
        )
        self.carbon_path = dp.get(
            "carbon_output_path", "data/carbon/carbon_intensity.json"
        )

        # Compatibility aliases for older code
        self.kafka_bootstrap = self.bootstrap_servers
        self.kafka_group_id = kafka.get("group_id", "velox-consumers")
        self.poll_timeout = self.poll_timeout_seconds
        self.max_per_poll = self.max_messages_per_poll
        self.pipeline_push_interval = self.pipeline_metrics_push_interval
        self.decision_window = self.decision_window_seconds

        if env_bootstrap:
            logger.info(
                "BridgeConfig: using CLOUDOS_KAFKA_BOOTSTRAP override=%s",
                env_bootstrap,
            )

    @classmethod
    def from_yaml(cls, path: str = "config/settings.yaml") -> "BridgeConfig":
        try:
            with open(path, encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}
            logger.info("BridgeConfig: loaded from %s", path)
        except FileNotFoundError:
            logger.warning("BridgeConfig: %s not found — using defaults.", path)
            raw = {}
        except Exception as exc:
            logger.warning("BridgeConfig: load error (%s) — using defaults.", exc)
            raw = {}

        merged: Dict[str, Any] = {}
        for section, defaults in _DEFAULTS.items():
            raw_section = raw.get(section, {}) or {}
            if isinstance(defaults, dict) and isinstance(raw_section, dict):
                merged[section] = {**defaults, **raw_section}
            else:
                merged[section] = raw_section or defaults

        for k, v in raw.items():
            if k not in merged:
                merged[k] = v

        return cls(merged)

    def raw(self) -> Dict[str, Any]:
        return dict(self._raw)