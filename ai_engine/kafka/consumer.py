"""
Velox Kafka Consumer
=======================

Base callback consumer + optional standalone metrics consumer.

Existing usage remains supported:
    consumer = VeloxConsumer(config, group_id="my-group", topics=["velox.alerts"])
    consumer.on("velox.alerts", handle_alert)
    consumer.start()

New standalone usage:
    python -m ai_engine.kafka.consumer

Standalone mode consumes velox.scheduling.decisions and exposes metrics at:
    http://localhost:9094/metrics
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional

import yaml

try:
    from confluent_kafka import Consumer, KafkaError, KafkaException
except ImportError:
    raise ImportError(
        "\n\nconfluent-kafka is not installed.\n"
        "Fix: run   pip install confluent-kafka   then try again.\n"
    )

try:
    from prometheus_client import CollectorRegistry, Counter, Gauge, start_http_server
except ImportError:
    CollectorRegistry = None
    Counter = None
    Gauge = None
    start_http_server = None

logger = logging.getLogger(__name__)

_METRICS_PORT = int(os.environ.get("CONSUMER_METRICS_PORT", "9094"))
_WINDOW_SIZE = 100


class VeloxConsumer:
    """
    Simple callback-based Kafka consumer.
    Each topic maps to one handler function via .on(topic, handler).
    """

    def __init__(self, config: Dict, group_id: str, topics: List[str]):
        kafka_cfg = config.get("kafka", {}) or {}
        servers = (
            os.environ.get("CLOUDOS_KAFKA_BOOTSTRAP", "").strip()
            or kafka_cfg.get("bootstrap_servers", "")
        )

        self._consumer = Consumer(
            {
                "bootstrap.servers": servers,
                "group.id": group_id,
                "auto.offset.reset": "latest",
                "enable.auto.commit": False,
                "session.timeout.ms": 30_000,
                "max.poll.interval.ms": 300_000,
            }
        )
        self._topics = topics
        self._handlers: Dict[str, Callable[[Dict], None]] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def on(self, topic: str, handler: Callable[[Dict], None]) -> "VeloxConsumer":
        """Register a handler for a topic. Returns self for chaining."""
        self._handlers[topic] = handler
        return self

    def start(self) -> None:
        """Start consuming in a background daemon thread."""
        self._consumer.subscribe(self._topics)
        self._running = True
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name=f"consumer-{'-'.join(self._topics[:2])}",
        )
        self._thread.start()
        logger.info("VeloxConsumer started — topics=%s", self._topics)

    def stop(self) -> None:
        """Signal the consumer to stop and wait for it."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=8.0)
        self._consumer.close()
        logger.info("VeloxConsumer stopped.")

    def _loop(self) -> None:
        while self._running:
            try:
                msg = self._consumer.poll(timeout=1.0)
                if msg is None:
                    continue

                if msg.error():
                    if msg.error().code() != KafkaError._PARTITION_EOF:
                        raise KafkaException(msg.error())
                    continue

                topic = msg.topic()
                try:
                    payload = json.loads(msg.value().decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    logger.warning("Consumer parse error [%s]: %s", topic, exc)
                    continue

                if topic in self._handlers:
                    try:
                        self._handlers[topic](payload)
                    except Exception as exc:
                        logger.exception("Handler error [%s]: %s", topic, exc)

                self._consumer.commit(asynchronous=False)

            except KafkaException as exc:
                logger.error("Consumer KafkaException: %s", exc)
                time.sleep(1.0)
            except Exception as exc:
                logger.exception("Consumer unexpected error: %s", exc)
                time.sleep(1.0)


class VeloxMetricsConsumer:
    """
    Standalone metrics consumer for velox.scheduling.decisions.
    Exposes rolling metrics on an HTTP endpoint for Prometheus scraping.
    """

    def __init__(self, config: Dict):
        kafka_cfg = config.get("kafka", {}) or {}
        topics_cfg = kafka_cfg.get("topics", {}) or {}

        self._bootstrap = (
            os.environ.get("CLOUDOS_KAFKA_BOOTSTRAP", "").strip()
            or kafka_cfg.get("bootstrap_servers", "")
        )
        self._topic = topics_cfg.get("decisions", "velox.scheduling.decisions")
        self._running = False

        self._consumer = Consumer(
            {
                "bootstrap.servers": self._bootstrap,
                "group.id": "velox-metrics-consumer",
                "auto.offset.reset": "latest",
                "enable.auto.commit": True,
                "session.timeout.ms": 30_000,
                "max.poll.interval.ms": 300_000,
            }
        )

        self._latencies: Deque[float] = deque(maxlen=_WINDOW_SIZE)
        self._cost_savings: Deque[float] = deque(maxlen=_WINDOW_SIZE)
        self._carbon_savings: Deque[float] = deque(maxlen=_WINDOW_SIZE)

        self._registry = None
        self._metrics = None
        self._init_metrics()

    def _init_metrics(self) -> None:
        if CollectorRegistry is None:
            logger.warning("prometheus_client not installed; metrics endpoint disabled")
            return

        self._registry = CollectorRegistry()
        self._metrics = {
            "decisions_total": Counter(
                "velox_consumer_decisions_total",
                "Total decisions consumed from Kafka",
                registry=self._registry,
            ),
            "latency_avg": Gauge(
                "velox_consumer_latency_ms_avg",
                "Rolling average latency in milliseconds",
                registry=self._registry,
            ),
            "cost_savings_avg": Gauge(
                "velox_consumer_cost_savings_avg_pct",
                "Rolling average cost savings percentage",
                registry=self._registry,
            ),
            "carbon_savings_avg": Gauge(
                "velox_consumer_carbon_savings_avg_pct",
                "Rolling average carbon savings percentage",
                registry=self._registry,
            ),
            "by_cloud": Counter(
                "velox_consumer_decisions_by_cloud_total",
                "Total decisions consumed by cloud provider",
                ["cloud"],
                registry=self._registry,
            ),
        }

    def start(self) -> None:
        if start_http_server is not None and self._registry is not None:
            start_http_server(_METRICS_PORT, registry=self._registry)
            logger.info(
                "Metrics consumer: Prometheus endpoint started at http://localhost:%d/metrics",
                _METRICS_PORT,
            )

        self._consumer.subscribe([self._topic])
        self._running = True
        logger.info(
            "Metrics consumer: subscribed to topic=%s bootstrap=%s",
            self._topic,
            self._bootstrap,
        )
        self._loop()

    def stop(self) -> None:
        self._running = False
        try:
            self._consumer.close()
        except Exception:
            pass
        logger.info("Metrics consumer stopped.")

    def _loop(self) -> None:
        while self._running:
            try:
                msg = self._consumer.poll(timeout=1.0)
                if msg is None:
                    continue

                if msg.error():
                    if msg.error().code() != KafkaError._PARTITION_EOF:
                        logger.warning("Metrics consumer poll error: %s", msg.error())
                    continue

                self._process_message(msg.value())

            except KafkaException as exc:
                logger.error("Metrics consumer KafkaException: %s", exc)
                time.sleep(1.0)
            except Exception as exc:
                logger.exception("Metrics consumer unexpected error: %s", exc)
                time.sleep(1.0)

    def _process_message(self, raw_value: bytes) -> None:
        try:
            payload = json.loads(raw_value.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.warning("Metrics consumer parse error: %s", exc)
            return

        latency = float(payload.get("latency_ms", 0.0) or 0.0)
        cost_savings = float(payload.get("cost_savings_pct", 0.0) or 0.0)
        carbon_savings = float(payload.get("carbon_savings_pct", 0.0) or 0.0)
        cloud = str(payload.get("cloud", "unknown"))
        decision_id = str(payload.get("decision_id", "?"))[:8]

        self._latencies.append(latency)
        self._cost_savings.append(cost_savings)
        self._carbon_savings.append(carbon_savings)

        if self._metrics is not None:
            self._metrics["decisions_total"].inc()
            self._metrics["latency_avg"].set(sum(self._latencies) / len(self._latencies))
            self._metrics["cost_savings_avg"].set(
                sum(self._cost_savings) / len(self._cost_savings)
            )
            self._metrics["carbon_savings_avg"].set(
                sum(self._carbon_savings) / len(self._carbon_savings)
            )
            self._metrics["by_cloud"].labels(cloud=cloud).inc()

        logger.info(
            "Metrics consumer: decision=%s cloud=%s latency=%.2fms cost=%.2f%% carbon=%.2f%%",
            decision_id,
            cloud,
            latency,
            cost_savings,
            carbon_savings,
        )


def _load_config() -> Dict:
    config_path = Path("config/settings.yaml")
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-40s  %(levelname)-5s  %(message)s",
    )

    consumer = VeloxMetricsConsumer(_load_config())

    def _shutdown(_sig, _frame) -> None:
        logger.info("Shutting down metrics consumer")
        consumer.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    consumer.start()


if __name__ == "__main__":
    main()