"""
Velox Kubernetes Operator
==============================
Controller loop that watches CloudWorkload custom resources
and drives the RL scheduling decision pipeline.

Architecture:
  ┌─────────────────────────────────────────────────────┐
  │  VeloxOperator (main loop)                          │
  │                                                     │
  │  poll_loop()  ←──── runs every poll_interval_sec    │
  │      │                                              │
  │      ├── list_pending()  ←── kubectl get cw         │
  │      │       phase in (Pending, "")                 │
  │      │                                              │
  │      └── for each pending workload:                 │
  │              set_phase(Scheduling)                  │
  │              workload = WorkloadMapper.map(cr)      │
  │              decision = SchedulerAgent.decide()     │
  │              decision_id = uuid4()                  │
  │              kafka_producer.publish_decision()      │
  │              set_scheduled(decision)                │
  └─────────────────────────────────────────────────────┘

Operator modes:
  --dry-run      logs decisions but does not patch CR status or send to Kafka
  --no-kafka     makes decisions + patches status, but skips Kafka publishing
  --no-shap      skips SHAP explainability (faster startup, no explanation field)

Compatible with:
  Module A — SchedulerAgent.decide() + SHAP explanation
  Module D — VeloxProducer.publish_decision()
  Module F — CloudWorkload CRD schema
"""

import json
import logging
import subprocess
import time
import uuid
from typing import Dict, List

from ai_engine.operator.lifecycle import WorkloadLifecycleManager
from ai_engine.operator.status_writer import StatusWriter
from ai_engine.operator.workload_mapper import WorkloadMapper

logger = logging.getLogger(__name__)

_DEFAULT_POLL_SEC = 5
_DEFAULT_NAMESPACE = "velox"
_DECISION_TIMEOUT_MS = 500  # warn if decision takes longer


class VeloxOperator:
    """
    Kubernetes operator controller loop for Velox.

    Watches CloudWorkload CRs in the configured namespace.
    For each workload in phase=Pending, calls the RL agent
    and patches the CR status with the scheduling decision.
    """

    def __init__(
        self,
        config: Dict,
        dry_run: bool = False,
        no_kafka: bool = False,
        no_shap: bool = False,
        namespace: str = _DEFAULT_NAMESPACE,
        poll_interval: int = _DEFAULT_POLL_SEC,
    ):
        self._config = config
        self._dry_run = dry_run
        self._no_kafka = no_kafka
        self._namespace = namespace
        self._poll_interval = poll_interval

        self._mapper = WorkloadMapper()
        self._writer = StatusWriter(dry_run=dry_run)
        self._lifecycle = WorkloadLifecycleManager(namespace, dry_run)

        self._agent = None
        self._producer = None

        self._no_shap = no_shap
        self._stats = {"processed": 0, "errors": 0, "skipped": 0}
        self._seen_rv: Dict[str, str] = {}  # name -> resourceVersion

        logger.info(
            "VeloxOperator: namespace=%s poll=%ds dry_run=%s no_kafka=%s no_shap=%s",
            namespace,
            poll_interval,
            dry_run,
            no_kafka,
            no_shap,
        )

    # -----------------------------------------------------------------------
    # Public — lifecycle
    # -----------------------------------------------------------------------

    def start(self):
        """
        Starts the operator controller loop.
        Blocks indefinitely. Use Ctrl+C to stop.
        """
        logger.info("VeloxOperator: loading agent ...")
        self._agent = self._load_agent()

        if not self._no_kafka:
            logger.info("VeloxOperator: connecting Kafka producer ...")
            self._producer = self._load_producer()

        logger.info(
            "VeloxOperator: entering poll loop (every %ds) ...",
            self._poll_interval,
        )
        logger.info(
            "VeloxOperator: watching namespace '%s' for CloudWorkloads ...",
            self._namespace,
        )

        try:
            while True:
                try:
                    self._poll_once()
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    logger.error("VeloxOperator: poll error: %s", exc, exc_info=True)
                time.sleep(self._poll_interval)
        except KeyboardInterrupt:
            logger.info("VeloxOperator: received shutdown signal.")
        finally:
            self._shutdown()

    def run_once(self) -> int:
        """
        Runs a single poll cycle.
        Useful for testing without starting the full loop.
        Returns the number of workloads processed.
        """
        if self._agent is None:
            self._agent = self._load_agent()
        if self._producer is None and not self._no_kafka:
            self._producer = self._load_producer()
        return self._poll_once()

    # -----------------------------------------------------------------------
    # Private — poll loop
    # -----------------------------------------------------------------------

    def _poll_once(self) -> int:
        """
        Lists all Pending CloudWorkloads and processes each one.
        Returns the count of workloads processed this cycle.
        Also ticks the lifecycle manager once per poll.
        """
        pending = self._list_pending()
        count = 0

        if pending:
            logger.info("VeloxOperator: found %d pending workload(s)", len(pending))

            for cr in pending:
                name = cr.get("metadata", {}).get("name", "unknown")
                rv = cr.get("metadata", {}).get("resourceVersion", "")

                if self._seen_rv.get(name) == rv:
                    self._stats["skipped"] += 1
                    continue

                success = self._process(cr)
                if success:
                    self._seen_rv[name] = rv
                    self._stats["processed"] += 1
                    count += 1
                else:
                    self._stats["errors"] += 1

        self._lifecycle.tick()
        return count

    def _process(self, cr: Dict) -> bool:
        """
        Processes a single CloudWorkload CR through the full pipeline:
          1. Patch phase=Scheduling
          2. Map CR spec -> workload dict
          2. RL agent decide()
          3. Kafka publish
          3. Patch phase=Scheduled with decision

        Returns True on success.
        """
        meta = cr.get("metadata", {})
        name = meta.get("name", "unknown")
        namespace = meta.get("namespace", self._namespace)

        logger.info("VeloxOperator: processing workload '%s/%s'", namespace, name)

        self._writer.set_scheduling(name, namespace)

        workload = self._mapper.map(cr)
        if workload is None:
            reason = f"WorkloadMapper failed to parse spec for '{name}'"
            logger.error("VeloxOperator: %s", reason)
            self._writer.set_failed(name, namespace, reason)
            return False

        t0 = time.perf_counter()
        try:
            decision = self._make_decision(workload)
        except Exception as exc:
            reason = f"RL agent error: {exc}"
            logger.error("VeloxOperator: %s", reason, exc_info=True)
            self._writer.set_failed(name, namespace, reason)
            return False

        latency_ms = (time.perf_counter() - t0) * 1000
        if latency_ms > _DECISION_TIMEOUT_MS:
            logger.warning(
                "VeloxOperator: decision for '%s' took %.0fms (target <500ms)",
                name,
                latency_ms,
            )

        decision_id = str(uuid.uuid4())
        decision["decision_id"] = decision_id
        decision["workload_id"] = name

        logger.info(
            "VeloxOperator: decision for '%s' -> %s/%s %s (cost=%.4f/hr, savings=%.1f%%, %.0fms)",
            name,
            decision.get("cloud"),
            decision.get("region"),
            decision.get("purchase_option"),
            decision.get("estimated_cost_per_hr", 0),
            decision.get("cost_savings_pct", 0),
            latency_ms,
        )

        if not self._no_kafka and self._producer is not None:
            self._publish(decision, workload)

        ok = self._writer.set_scheduled(name, namespace, decision)
        if not ok:
            logger.error(
                "VeloxOperator: status patch failed for '%s' (decision was still made)",
                name,
            )
            return False

        return True

    # -----------------------------------------------------------------------
    # Private — decision
    # -----------------------------------------------------------------------

    def _make_decision(self, workload: Dict) -> Dict:
        """
        Calls SchedulerAgent.decide() if model is loaded.
        Falls back to a deterministic heuristic if model not available.
        """
        if self._agent is not None:
            decision = self._agent.decide(workload)
            if decision:
                return decision
            logger.warning(
                "VeloxOperator: agent returned None — using heuristic fallback"
            )

        return self._heuristic_decision(workload)

    def _heuristic_decision(self, workload: Dict) -> Dict:
        """
        Deterministic fallback used when the PPO model is not yet trained.
        Picks eu-north-1 on spot if spot-tolerant,
        otherwise us-east-1 on-demand.
        """
        spot_ok = bool(workload.get("is_spot_tolerant", 0))
        cloud = "aws"
        region = "eu-north-1" if spot_ok else "us-east-1"
        purchase = "spot" if spot_ok else "on_demand"

        return {
            "cloud": cloud,
            "region": region,
            "instance_type": "m5.large",
            "purchase_option": purchase,
            "sla_tier": workload.get("sla_tier", "standard"),
            "estimated_cost_per_hr": 0.032 if spot_ok else 0.096,
            "cost_savings_pct": 66.7 if spot_ok else 0.0,
            "carbon_savings_pct": 89.9 if spot_ok else 0.0,
            "latency_ms": 1.5,
            "explanation": {
                "summary": (
                    f"Heuristic decision: {cloud}/{region} ({purchase}). "
                    "RL model not yet trained."
                ),
                "top_drivers": [],
                "confidence": 0.0,
            },
        }

    # -----------------------------------------------------------------------
    # Private — Kafka
    # -----------------------------------------------------------------------

    def _publish(self, decision: Dict, workload: Dict):
        """Publishes decision to Kafka. Swallows errors to avoid blocking the operator."""
        try:
            self._producer.publish_decision(
                {
                    **decision,
                    "workload_type": workload.get("workload_type", "batch"),
                }
            )
            logger.debug(
                "VeloxOperator: published decision %s to Kafka",
                decision.get("decision_id"),
            )
        except Exception as exc:
            logger.warning(
                "VeloxOperator: Kafka publish failed (non-fatal): %s",
                exc,
            )

    # -----------------------------------------------------------------------
    # Private — list pending CRs
    # -----------------------------------------------------------------------

    def _list_pending(self) -> List[Dict]:
        """
        Lists CloudWorkload CRs in phase=Pending (or phase unset).
        Uses kubectl get cloudworkloads -o json.
        """
        cmd = [
            "kubectl",
            "get",
            "cloudworkloads",
            "-n",
            self._namespace,
            "-o",
            "json",
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                logger.warning(
                    "VeloxOperator: kubectl get failed: %s",
                    result.stderr.strip()[:200],
                )
                return []

            data = json.loads(result.stdout)
            items = data.get("items", [])

            pending = []
            for item in items:
                phase = item.get("status", {}).get("phase", "")
                if phase in ("", "Pending"):
                    pending.append(item)

            return pending

        except subprocess.TimeoutExpired:
            logger.warning("VeloxOperator: kubectl timed out")
            return []
        except json.JSONDecodeError as exc:
            logger.warning("VeloxOperator: JSON parse error: %s", exc)
            return []
        except Exception as exc:
            logger.error("VeloxOperator: list_pending error: %s", exc)
            return []

    # -----------------------------------------------------------------------
    # Private — loader helpers
    # -----------------------------------------------------------------------

    def _load_agent(self):
        """
        Loads SchedulerAgent. Returns None if model not found.
        Operator continues in heuristic mode if model not yet trained.
        """
        try:
            from ai_engine.inference.scheduler_agent import SchedulerAgent

            agent = SchedulerAgent.load(
                config=self._config,
                with_explainer=(not self._no_shap),
            )
            if agent:
                logger.info("VeloxOperator: SchedulerAgent loaded (PPO + SHAP mode)")
            else:
                logger.warning(
                    "VeloxOperator: SchedulerAgent.load returned None "
                    "(model not trained yet) — using heuristic fallback"
                )
            return agent
        except Exception as exc:
            logger.warning(
                "VeloxOperator: agent load failed (%s) — using heuristic fallback",
                exc,
            )
            return None

    def _load_producer(self):
        """Loads KafkaProducer. Returns None if Kafka not reachable."""
        try:
            from ai_engine.kafka.producer import VeloxProducer

            producer = VeloxProducer(self._config)
            logger.info("VeloxOperator: Kafka producer ready")
            return producer
        except Exception as exc:
            logger.warning(
                "VeloxOperator: Kafka producer load failed (%s) — decisions will not be published",
                exc,
            )
            return None

    def _shutdown(self):
        """Clean shutdown."""
        if self._producer is not None:
            try:
                self._producer.flush()
            except Exception:
                pass

        logger.info(
            "VeloxOperator: shutdown — processed=%d errors=%d skipped=%d",
            self._stats["processed"],
            self._stats["errors"],
            self._stats["skipped"],
        )