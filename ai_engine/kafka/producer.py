# FULL ORIGINAL FILE WITH SAFE PATCHES APPLIED
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from confluent_kafka import Producer as KafkaProducer
except ImportError:
    KafkaProducer = None

# IMPORTANT FIX: expose Producer for tests
Producer = KafkaProducer

TOPICS = {
    "decisions": "velox.scheduling.decisions",
    "metrics": "velox.metrics",
    "alerts": "velox.alerts",
    "workloads": "velox.workload.events",
}

_REQUIRED_DECISION_FIELDS = frozenset(
    ["decision_id", "cloud", "region", "instance_type", "purchase_option"]
)

_FLUSH_INTERVAL_SEC = 5


class VeloxProducer:
    """
    Thread-safe Kafka producer for Velox.

    Goals:
    - Never break API inference flow if Kafka is unavailable
    - Be safe in Kubernetes/Minikube environments
    - Preserve existing topic names and public methods
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        bootstrap_servers: Optional[str] = None,
    ):
        if isinstance(config, str) and bootstrap_servers is None:
            bootstrap_servers = config
            config = {}

        config = config or {}
        self._config = config
        self._lock = threading.Lock()

        kafka_cfg = config.get("kafka", {}) or {}

        if bootstrap_servers:
            self._servers = bootstrap_servers
            logger.info(
                "VeloxProducer: bootstrap_servers=%s (source: explicit argument)",
                self._servers,
            )
        else:
            self._servers = self._resolve_bootstrap(kafka_cfg)

        self._topics = dict(TOPICS)

        self._partitions = int(kafka_cfg.get("partitions", 3))
        self._replication = int(kafka_cfg.get("replication", 1))

        self._producer = None

        logger.info("VeloxProducer: bootstrap_servers=%s", self._servers if self._servers else "disabled")

        # Only connect if we have valid servers
        if self._servers:
            self._connect()
            self._ensure_topics()
        else:
            logger.info("Kafka disabled — no bootstrap servers configured")

        self._flush_thread = threading.Thread(
            target=self._flush_loop,
            daemon=True,
            name="kafka-flush",
        )
        self._flush_thread.start()

    # ---------------------------------------------------------------------
    # Bootstrap / connection
    # ---------------------------------------------------------------------

    @staticmethod
    def _resolve_bootstrap(config: Dict[str, Any]) -> str:
        sources = [
            (
                "env:CLOUDOS_KAFKA_BOOTSTRAP",
                os.environ.get("CLOUDOS_KAFKA_BOOTSTRAP", "").strip(),
            ),
            (
                "settings.yaml:kafka.bootstrap",
                str(config.get("bootstrap_servers", "") or "").strip(),
            ),
        ]

        for source_name, candidate in sources:
            if not candidate:
                logger.debug(
                    "VeloxProducer._resolve_bootstrap: %s is empty — skipping",
                    source_name,
                )
                continue

            lowered = candidate.lower()
            if "localhost" in lowered or "127.0.0.1" in lowered or "192.168." in lowered or "10." in lowered or "172.16." in lowered:
                logger.warning(
                    "VeloxProducer._resolve_bootstrap: %s='%s' contains private/local IP — skipping",
                    source_name,
                    candidate,
                )
                continue

            logger.info(
                "VeloxProducer: bootstrap_servers=%s (source: %s)",
                candidate,
                source_name,
            )
            return candidate

        logger.info("VeloxProducer: no valid bootstrap servers found — Kafka disabled")
        return ""

    def _connect(self) -> None:
        if Producer is None:
            logger.warning("Kafka disabled (confluent_kafka missing)")
            self._producer = None
            return

        if not self._servers:
            logger.info("Kafka disabled — no valid bootstrap servers configured")
            self._producer = None
            return

        try:
            producer = Producer(
                {
                    "bootstrap.servers": self._servers,
                    "client.id": "velox-producer",
                    "acks": "all",
                    "retries": 3,
                    "retry.backoff.ms": 500,
                    "compression.type": "lz4",
                    "linger.ms": 5,
                    "message.max.bytes": 1_048_576,
                    "socket.timeout.ms": 6000,
                    "message.timeout.ms": 15000,
                    "socket.connection.setup.timeout.ms": 6000,
                    "log_level": 3,
                }
            )

            if self._probe_connection(producer):
                with self._lock:
                    self._producer = producer
                logger.info("Kafka connected at %s", self._servers)
            else:
                self._producer = None
                logger.warning("Kafka not reachable")

        except Exception as exc:
            self._producer = None
            logger.warning("Kafka connection failed: %s", exc)

    @staticmethod
    def _probe_connection(producer: Any, timeout: float = 4.0) -> bool:
        try:
            meta = producer.list_topics(timeout=timeout)
            return bool(getattr(meta, "brokers", {}))
        except Exception:
            return False

    def _reconnect_if_needed(self) -> bool:
        with self._lock:
            if self._producer is not None:
                return True

        # Don't attempt reconnect if Kafka is disabled
        if not self._servers:
            return False

        logger.info("Kafka reconnect attempt")
        self._connect()

        with self._lock:
            return self._producer is not None

    # ---------------------------------------------------------------------
    # Publish methods
    # ---------------------------------------------------------------------

    def publish_decision(self, decision: Dict[str, Any]) -> bool:
        missing = _REQUIRED_DECISION_FIELDS - set(decision.keys())
        if missing:
            logger.warning("Missing fields: %s", missing)
            return False

        if not self._reconnect_if_needed():
            return False

        raw_key = decision.get("decision_id")
        key_str = str(raw_key) if raw_key else str(int(time.time() * 1000))
        encoded_key = key_str.encode("utf-8")
        topic = self._topics["decisions"]

        payload = {
            **decision,
            "ts": time.time(),
            "ts_iso": _now_iso(),
        }

        encoded_value = json.dumps(payload, default=str).encode("utf-8")

        try:
            with self._lock:
                producer = self._producer
                if producer is None:
                    return False

                producer.produce(
                    topic=topic,
                    key=encoded_key,
                    value=encoded_value,
                    on_delivery=self._on_delivery,
                )

            # 🚀 NON-BLOCKING FIX
            producer.poll(0.5)

            return True

        except BufferError:
            logger.warning("Buffer full — polling instead of flush")
            try:
                producer.poll(0.5)
            except Exception:
                pass
            return False

        except Exception as exc:
            logger.warning("Publish failed: %s", exc)
            return False

    # ---------------- REMAINING METHODS UNCHANGED ----------------

    def publish_metrics(self, metrics: Dict[str, Any]) -> bool:
        key = str(int(time.time() * 1000))
        payload = {**metrics, "_ts": time.time(), "_ts_iso": _now_iso()}
        return self._send(self._topics["metrics"], key, payload)

    def publish_alert(self, kind: str, detail: Dict[str, Any]) -> bool:
        payload = {
            "kind": kind,
            "detail": detail,
            "ts": time.time(),
            "ts_iso": _now_iso(),
        }
        return self._send(self._topics["alerts"], kind, payload)

    def publish_workload_event(self, workload_id, workload_type, event_type, detail=None):
        payload = {
            "workload_id": workload_id,
            "workload_type": workload_type,
            "event_type": event_type,
            "detail": detail or {},
            "ts": time.time(),
            "ts_iso": _now_iso(),
        }
        return self._send(self._topics["workloads"], workload_id, payload)

    def _send(self, topic, key, payload):
        if not self._reconnect_if_needed():
            return False

        try:
            encoded_key = str(key).encode() if key else None
            encoded_value = json.dumps(payload, default=str).encode()

            with self._lock:
                producer = self._producer
                if producer is None:
                    return False

                producer.produce(
                    topic=topic,
                    key=encoded_key,
                    value=encoded_value,
                    on_delivery=self._on_delivery,
                )

            producer.poll(0)
            return True

        except Exception:
            return False

    def flush(self, timeout: float = 5.0):
        with self._lock:
            producer = self._producer
        if producer:
            producer.flush(timeout)

    def _ensure_topics(self):
        try:
            from confluent_kafka.admin import AdminClient, NewTopic

            admin = AdminClient({"bootstrap.servers": self._servers})
            existing = set(admin.list_topics(timeout=5).topics)

            new_topics = [
                NewTopic(t, self._partitions, self._replication)
                for t in self._topics.values()
                if t not in existing
            ]

            if new_topics:
                admin.create_topics(new_topics)

        except Exception:
            pass

    def _flush_loop(self):
        while True:
            time.sleep(_FLUSH_INTERVAL_SEC)
            with self._lock:
                producer = self._producer
            if producer:
                try:
                    producer.poll(0)
                except Exception:
                    pass

    @staticmethod
    def _on_delivery(err, msg):
        if err:
            logger.warning("Kafka delivery failed: %s", err)


def _now_iso():
    return datetime.now(timezone.utc).isoformat()