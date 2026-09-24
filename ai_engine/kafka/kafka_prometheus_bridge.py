"""
Kafka → Prometheus Bridge
=========================
Consumes messages from three Kafka topics and exposes them
as Prometheus metrics at http://0.0.0.0:9090/metrics.

Topics consumed:
  velox.scheduling.decisions  → cost, carbon, latency, cloud, region metrics
  velox.metrics               → general system metrics
  velox.alerts                → alert counters by kind

Prometheus metrics exposed (full list in metrics_registry.py):
  velox_decisions_total
  velox_inference_latency_seconds
  velox_cost_savings_ratio
  velox_carbon_savings_ratio
  velox_rl_reward
  velox_alerts_total
  velox_bridge_messages_consumed_total
  velox_bridge_up
  velox_carbon_intensity_gco2_per_kwh   (from pipeline data files)
  velox_pricing_on_demand_usd_per_hr    (from pipeline data files)
  ... (full list in metrics_registry.py)

Thread model:
  MainThread       → Prometheus HTTP server (blocking serve_forever)
  consumer-thread  → Kafka poll loop (daemon thread)
  pipeline-thread  → pushes pipeline file metrics every N seconds (daemon)

Graceful shutdown:
  Ctrl+C → sets _running=False → consumer commits offsets → threads join

CLI:
  python -m ai_engine.kafka.kafka_prometheus_bridge
  python -m ai_engine.kafka.kafka_prometheus_bridge --port 9091

Compatible with:
  - confluent-kafka Python client
  - prometheus_client HTTP server
  - Module G data pipeline (reads carbon + pricing JSON files)
  - Module G Kafka producer (reads messages it publishes)
"""

from __future__ import annotations

import json
import logging
import signal
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Optional

# ---------------------------------------------------------------------------
# Dependency checks with clear error messages
# ---------------------------------------------------------------------------
try:
    from confluent_kafka import Consumer, KafkaError, KafkaException
    from confluent_kafka.admin import AdminClient
except ImportError:
    raise ImportError(
        "\n\nconfluent-kafka is not installed.\n"
        "Fix: run   pip install confluent-kafka   then try again.\n"
    )

try:
    from prometheus_client import start_http_server
except ImportError:
    raise ImportError(
        "\n\nprometheus-client is not installed.\n"
        "Fix: run   pip install prometheus-client   then try again.\n"
    )

from ai_engine.kafka.bridge_config import BridgeConfig
from ai_engine.kafka.metrics_registry import (
    ACTIVE_DECISIONS,
    ALERTS_TOTAL,
    BRIDGE_MESSAGES_CONSUMED,
    BRIDGE_PARSE_ERRORS,
    BRIDGE_UP,
    CARBON_INTENSITY_GAUGE,
    CARBON_SAVINGS_RATIO,
    COST_SAVINGS_RATIO,
    DECISIONS_TOTAL,
    ESTIMATED_COST_PER_HR,
    INFERENCE_LATENCY,
    LAST_DECISION_TIMESTAMP,
    PIPELINE_CARBON_FETCHES,
    PIPELINE_CUR_FETCHES,
    PIPELINE_ERRORS,
    PIPELINE_PRICING_FETCHES,
    PRICING_ON_DEMAND_GAUGE,
    RL_REWARD,
    WORKLOAD_EVENTS_TOTAL,
)

logger = logging.getLogger(__name__)

# Kafka topic names — must match producer.py TOPICS dict
TOPIC_DECISIONS = "velox.scheduling.decisions"
TOPIC_METRICS = "velox.metrics"
TOPIC_ALERTS = "velox.alerts"
TOPIC_WORKLOADS = "velox.workload.events"

ALL_TOPICS = [TOPIC_DECISIONS, TOPIC_METRICS, TOPIC_ALERTS, TOPIC_WORKLOADS]


class KafkaPrometheusBridge:
    """
    Kafka consumer that translates messages into Prometheus metrics.
    Runs Prometheus HTTP server in the main thread and Kafka consumer
    in a background daemon thread.
    """

    def __init__(self, config: BridgeConfig):
        self._config = config
        self._running = False

        # Rolling window for decisions-per-minute gauge
        self._decision_timestamps: Deque[float] = deque()
        self._decision_lock = threading.Lock()

        # Thread references
        self._consumer_thread: Optional[threading.Thread] = None
        self._pipeline_thread: Optional[threading.Thread] = None
        self._gauge_thread: Optional[threading.Thread] = None

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def start(self) -> None:
        """
        Starts all background threads.
        Returns immediately — caller should call wait() or run the main loop.
        """
        self._running = True
        BRIDGE_UP.set(1)
        logger.info(
            "KafkaPrometheusBridge starting — Prometheus at http://%s:%d/metrics",
            self._config.prometheus_host,
            self._config.prometheus_port,
        )

        self._consumer_thread = threading.Thread(
            target=self._consumer_loop,
            daemon=True,
            name="kafka-consumer",
        )
        self._consumer_thread.start()

        self._pipeline_thread = threading.Thread(
            target=self._pipeline_metrics_loop,
            daemon=True,
            name="pipeline-metrics",
        )
        self._pipeline_thread.start()

        self._gauge_thread = threading.Thread(
            target=self._active_decisions_loop,
            daemon=True,
            name="gauge-updater",
        )
        self._gauge_thread.start()

        logger.info("KafkaPrometheusBridge: all threads started.")

    def stop(self) -> None:
        """Sets the shutdown flag. Consumer thread will exit on next poll cycle."""
        self._running = False
        BRIDGE_UP.set(0)
        logger.info("KafkaPrometheusBridge: stop signal sent.")

    def wait(self, timeout: float = 10.0) -> None:
        """Blocks until the consumer thread exits."""
        if self._consumer_thread:
            self._consumer_thread.join(timeout=timeout)

    def run_prometheus_server(self) -> None:
        """
        Starts the Prometheus HTTP server.
        BLOCKING — call this in the main thread after start().
        """
        start_http_server(
            port=self._config.prometheus_port,
            addr=self._config.prometheus_host,
        )
        logger.info(
            "Prometheus metrics server started at http://%s:%d/metrics",
            self._config.prometheus_host,
            self._config.prometheus_port,
        )

        while self._running:
            time.sleep(1.0)

    # -----------------------------------------------------------------------
    # Kafka consumer loop
    # -----------------------------------------------------------------------

    def _consumer_loop(self) -> None:
        """
        Main Kafka poll loop. Runs in background thread.
        Creates consumer, subscribes to all topics, polls continuously.
        """
        consumer = self._create_consumer()
        if consumer is None:
            logger.error(
                "KafkaPrometheusBridge: consumer creation failed. Thread exiting."
            )
            return

        try:
            consumer.subscribe(ALL_TOPICS)
            logger.info("Subscribed to topics: %s", ALL_TOPICS)

            while self._running:
                try:
                    msg = consumer.poll(timeout=self._config.poll_timeout)

                    if msg is None:
                        continue

                    if msg.error():
                        if msg.error().code() == KafkaError._PARTITION_EOF:
                            continue
                        logger.error("Kafka consumer error: %s", msg.error())
                        continue

                    self._handle_message(msg.topic(), msg.value())
                    consumer.commit(asynchronous=False)

                except KafkaException as exc:
                    logger.error("KafkaException in poll loop: %s", exc)
                    time.sleep(2.0)
                except Exception as exc:
                    logger.exception("Unexpected error in consumer loop: %s", exc)
                    time.sleep(1.0)

        finally:
            consumer.close()
            logger.info("Kafka consumer closed.")

    def _create_consumer(self) -> Optional[Consumer]:
        """Creates and returns the Kafka consumer. Returns None on failure."""
        conf = {
            "bootstrap.servers": self._config.kafka_bootstrap,
            "group.id": self._config.kafka_group_id + "-bridge",
            "auto.offset.reset": "latest",
            "enable.auto.commit": False,
            "session.timeout.ms": 30000,
            "max.poll.interval.ms": 300000,
            "fetch.min.bytes": 1,
            "fetch.wait.max.ms": 500,
        }

        max_retries = 3

        for attempt in range(1, max_retries + 1):
            try:
                admin = AdminClient({"bootstrap.servers": self._config.kafka_bootstrap})
                topics = admin.list_topics(timeout=5)

                logger.info(
                    "Kafka connection OK (attempt %d) — %d topics available.",
                    attempt,
                    len(topics.topics),
                )

                return Consumer(conf)

            except KafkaException as exc:
                logger.warning(
                    "Kafka connection attempt %d/%d failed: %s\n"
                    "  -> Make sure Kafka is running at %s",
                    attempt,
                    max_retries,
                    exc,
                    self._config.kafka_bootstrap,
                )

                if attempt < max_retries:
                    logger.info("Retrying in 5 seconds ...")
                    time.sleep(5)

        logger.error(
            "Kafka unreachable after %d attempts at %s.\n"
            "Prometheus server will still start but Kafka consumer will not run.",
            max_retries,
            self._config.kafka_bootstrap,
        )
        return None

    # -----------------------------------------------------------------------
    # Message handlers
    # -----------------------------------------------------------------------

    def _handle_message(self, topic: str, raw: bytes) -> None:
        """Dispatch message to the correct handler by topic."""
        BRIDGE_MESSAGES_CONSUMED.labels(topic=topic).inc()

        try:
            payload: Dict[str, Any] = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            BRIDGE_PARSE_ERRORS.labels(topic=topic).inc()
            logger.warning("Parse error [%s]: %s", topic, exc)
            return

        try:
            if topic == TOPIC_DECISIONS:
                self._handle_decision(payload)
            elif topic == TOPIC_METRICS:
                self._handle_metrics(payload)
            elif topic == TOPIC_ALERTS:
                self._handle_alert(payload)
            elif topic == TOPIC_WORKLOADS:
                self._handle_workload(payload)
        except Exception as exc:
            BRIDGE_PARSE_ERRORS.labels(topic=topic).inc()
            logger.warning("Handler error [%s]: %s | payload=%s", topic, exc, payload)

    def _handle_decision(self, d: Dict[str, Any]) -> None:
        """
        Handles velox.scheduling.decisions messages.
        Schema (produced by backend/api/routes/scheduling.py / producer.py):
          {
            decision_id, workload_id,
            cloud, region, instance_type,
            purchase_option, sla_tier,
            estimated_cost_per_hr, cost_savings_pct,
            carbon_savings_pct, latency_ms,
            explanation, actual_reward
          }
        """
        cloud = d.get("cloud", "unknown")
        region = d.get("region", "unknown")
        inst = d.get("instance_type", "unknown")
        purchase = d.get("purchase_option", "unknown")

        DECISIONS_TOTAL.labels(
            cloud=cloud,
            region=region,
            instance_type=inst,
            purchase_option=purchase,
        ).inc()

        latency_ms = float(d.get("latency_ms", 0.0))
        latency_sec = latency_ms / 1000.0
        INFERENCE_LATENCY.labels(cloud=cloud).observe(latency_sec)

        cost_savings_pct = float(d.get("cost_savings_pct", 0.0))
        COST_SAVINGS_RATIO.labels(
            cloud=cloud,
            purchase_option=purchase,
        ).observe(cost_savings_pct / 100.0)

        carbon_savings_pct = float(d.get("carbon_savings_pct", 0.0))
        CARBON_SAVINGS_RATIO.labels(
            cloud=cloud,
            region=region,
        ).observe(carbon_savings_pct / 100.0)

        cost_per_hr = float(d.get("estimated_cost_per_hr", 0.0))
        ESTIMATED_COST_PER_HR.labels(
            cloud=cloud,
            instance_type=inst,
        ).observe(cost_per_hr)

        reward = d.get("actual_reward")
        if reward is not None:
            try:
                RL_REWARD.labels(cloud=cloud).observe(float(reward))
            except (TypeError, ValueError):
                logger.warning("Invalid actual_reward value: %r", reward)

        LAST_DECISION_TIMESTAMP.set(time.time())

        with self._decision_lock:
            self._decision_timestamps.append(time.time())

        logger.debug(
            "Decision: cloud=%s region=%s cost_savings=%.1f%% carbon_savings=%.1f%% latency=%.1fms",
            cloud,
            region,
            cost_savings_pct,
            carbon_savings_pct,
            latency_ms,
        )

    def _handle_metrics(self, d: Dict[str, Any]) -> None:
        """
        Handles velox.metrics messages.
        Schema (produced by kafka/producer.py publish_metrics):
          {
            pricing_fetches, carbon_fetches, cur_fetches,
            pricing_errors, carbon_errors, cur_errors,
            ... any additional metric fields
          }
        """
        if "pricing_fetches" in d:
            PIPELINE_PRICING_FETCHES.set(float(d["pricing_fetches"]))
        if "carbon_fetches" in d:
            PIPELINE_CARBON_FETCHES.set(float(d["carbon_fetches"]))
        if "cur_fetches" in d:
            PIPELINE_CUR_FETCHES.set(float(d["cur_fetches"]))
        if "pricing_errors" in d:
            PIPELINE_ERRORS.labels(type="pricing").set(float(d["pricing_errors"]))
        if "carbon_errors" in d:
            PIPELINE_ERRORS.labels(type="carbon").set(float(d["carbon_errors"]))
        if "cur_errors" in d:
            PIPELINE_ERRORS.labels(type="cur").set(float(d["cur_errors"]))

        logger.debug("Metrics message processed.")

    def _handle_alert(self, d: Dict[str, Any]) -> None:
        """
        Handles velox.alerts messages.
        Schema (produced by kafka/producer.py publish_alert):
          {"kind": "cost_anomaly", "detail": {...}, "ts": float}
        """
        kind = d.get("kind", "unknown")
        ALERTS_TOTAL.labels(kind=kind).inc()
        logger.warning("ALERT received [%s]: %s", kind, d.get("detail", {}))

    def _handle_workload(self, d: Dict[str, Any]) -> None:
        """
        Handles velox.workload.events messages.
        Schema: {"workload_id", "workload_type", "event_type", ...}
        """
        workload_type = d.get("workload_type", "unknown")
        event_type = d.get("event_type", "unknown")
        WORKLOAD_EVENTS_TOTAL.labels(
            workload_type=workload_type,
            event_type=event_type,
        ).inc()

    # -----------------------------------------------------------------------
    # Pipeline file metrics pusher
    # -----------------------------------------------------------------------

    def _pipeline_metrics_loop(self) -> None:
        """
        Reads Module G output files and pushes regional carbon + pricing
        data as Prometheus gauges every N seconds.
        This runs even when no Kafka messages arrive.
        """
        while self._running:
            try:
                self._push_carbon_gauges()
                self._push_pricing_gauges()
            except Exception as exc:
                logger.warning("Pipeline metrics push error: %s", exc)

            for _ in range(self._config.pipeline_push_interval):
                if not self._running:
                    break
                time.sleep(1.0)

    def _push_carbon_gauges(self) -> None:
        """Reads carbon_intensity.json and sets per-region Prometheus gauges."""
        carbon_path = Path(self._config.carbon_path)
        if not carbon_path.exists():
            return

        try:
            with carbon_path.open(encoding="utf-8") as fh:
                data = json.load(fh)

            for region, entry in data.items():
                co2 = float(entry.get("gco2_per_kwh", 0.0))
                if co2 > 0:
                    CARBON_INTENSITY_GAUGE.labels(region=region).set(co2)
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
            logger.debug("Carbon gauge push error: %s", exc)

    def _push_pricing_gauges(self) -> None:
        """Reads aws_pricing.json and sets per-region m5.large on-demand gauge."""
        pricing_path = Path(self._config.pricing_path)
        if not pricing_path.exists():
            return

        try:
            with pricing_path.open(encoding="utf-8") as fh:
                data = json.load(fh)

            for region, entry in data.items():
                if region.startswith("_"):
                    continue
                price = entry.get("m5.large") or entry.get("m5.large:on_demand", 0.0)
                if price and float(price) > 0:
                    PRICING_ON_DEMAND_GAUGE.labels(region=region).set(float(price))
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
            logger.debug("Pricing gauge push error: %s", exc)

    # -----------------------------------------------------------------------
    # Active decisions gauge updater
    # -----------------------------------------------------------------------

    def _active_decisions_loop(self) -> None:
        """
        Updates the active_decisions gauge by counting decisions
        in the last decision_window seconds.
        Runs every 10 seconds.
        """
        while self._running:
            cutoff = time.time() - self._config.decision_window
            with self._decision_lock:
                while self._decision_timestamps and self._decision_timestamps[0] < cutoff:
                    self._decision_timestamps.popleft()
                count = len(self._decision_timestamps)

            ACTIVE_DECISIONS.set(count)
            time.sleep(10.0)


# ---------------------------------------------------------------------------
# Entry point: python -m ai_engine.kafka.kafka_prometheus_bridge
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-45s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    bridge_logger = logging.getLogger("bridge_main")

    port = 9090
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--port" and i + 1 < len(args):
            try:
                port = int(args[i + 1])
            except ValueError:
                pass

    config = BridgeConfig.from_yaml("config/settings.yaml")
    config.prometheus_port = port
    if "prometheus" not in config._raw:
        config._raw["prometheus"] = {}
    config._raw["prometheus"]["port"] = port

    bridge = KafkaPrometheusBridge(config)

    def _shutdown(sig, frame) -> None:
        bridge_logger.info("Shutdown signal — stopping bridge ...")
        bridge.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    bridge.start()
    bridge.run_prometheus_server()