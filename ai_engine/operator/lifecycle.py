"""
Workload Lifecycle Manager
===========================
Transitions CloudWorkload phases beyond Scheduled.

Phase state machine:
  Pending → Scheduling → Scheduled → Running → Completed
                                    ↘ Failed

Scheduled → Running:
  After a configurable delay (simulates actual cloud provisioning time).
  In production: triggered by cloud provider webhook or polling the jobs API.

Running → Completed:
  After expectedDurationHours has elapsed from scheduledAt time.
  In production: triggered by actual job completion events.
"""

import json
import logging
import subprocess
import time
from datetime import datetime, timezone, timedelta
from typing import List, Dict

logger = logging.getLogger(__name__)

_PROVISION_DELAY_SECONDS = 30   # simulate cloud provisioning time
_CHECK_INTERVAL_SECONDS  = 15


class WorkloadLifecycleManager:
    """
    Runs as a background thread inside the operator.
    Periodically checks Scheduled workloads and advances their phase.
    """

    def __init__(self, namespace: str = "velox", dry_run: bool = False):
        self._namespace = namespace
        self._dry_run   = dry_run
        self._scheduled_at: Dict[str, datetime] = {}

    def tick(self):
        """Call periodically from operator loop."""
        try:
            self._advance_scheduled_to_running()
            self._advance_running_to_completed()
        except Exception as exc:
            logger.error("WorkloadLifecycleManager.tick error: %s", exc)

    def _advance_scheduled_to_running(self):
        """Transitions Scheduled → Running after provision delay."""
        items = self._list_by_phase("Scheduled")
        now   = datetime.now(timezone.utc)

        for item in items:
            name         = item["metadata"]["name"]
            scheduled_at = self._parse_scheduled_at(item)

            if scheduled_at is None:
                continue

            elapsed = (now - scheduled_at).total_seconds()
            if elapsed >= _PROVISION_DELAY_SECONDS:
                logger.info(
                    "Lifecycle: %s Scheduled→Running (%.0fs elapsed)",
                    name, elapsed
                )
                self._patch_phase(name, "Running",
                                  f"Provisioned on {item['status'].get('scheduledCloud','?')}/"
                                  f"{item['status'].get('scheduledRegion','?')}.")

    def _advance_running_to_completed(self):
        """Transitions Running → Completed after expectedDurationHours."""
        items = self._list_by_phase("Running")
        now   = datetime.now(timezone.utc)

        for item in items:
            name         = item["metadata"]["name"]
            expected_hrs = float(item.get("spec", {}).get("expectedDurationHours", 1.0))
            scheduled_at = self._parse_scheduled_at(item)

            if scheduled_at is None:
                continue

            elapsed_hrs = (now - scheduled_at).total_seconds() / 3600
            if elapsed_hrs >= expected_hrs:
                logger.info(
                    "Lifecycle: %s Running→Completed (%.2f hrs)",
                    name, elapsed_hrs
                )
                self._patch_phase(name, "Completed",
                                  f"Completed after {elapsed_hrs:.2f}h on "
                                  f"{item['status'].get('scheduledCloud','?')}.")

    def _list_by_phase(self, phase: str) -> List[Dict]:
        cmd = ["kubectl", "get", "cloudworkloads", "-n", self._namespace, "-o", "json"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if r.returncode != 0:
                return []
            data = json.loads(r.stdout)
            return [i for i in data.get("items", [])
                    if i.get("status", {}).get("phase") == phase]
        except Exception:
            return []

    def _patch_phase(self, name: str, phase: str, message: str):
        from ai_engine.operator.status_writer import _now_iso
        patch = json.dumps({"status": {"phase": phase, "message": message,
                                       "scheduledAt": _now_iso()}})
        if self._dry_run:
            logger.info("[DRY RUN] patch %s → %s", name, phase)
            return
        cmd = ["kubectl", "patch", "cloudworkload", name,
               "-n", self._namespace, "--subresource=status",
               "--type=merge", "-p", patch]
        subprocess.run(cmd, capture_output=True, timeout=10)

    @staticmethod
    def _parse_scheduled_at(item: Dict):
        ts = item.get("status", {}).get("scheduledAt", "")
        if not ts:
            return None
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            return None