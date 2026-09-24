"""
Workload Mapper
================
Converts a CloudWorkload Kubernetes CR spec into the dict format
expected by SchedulerAgent.decide() and StateBuilder.build().

CloudWorkload CR spec (from crd.yaml):              SchedulerAgent workload dict:
─────────────────────────────────────────           ──────────────────────────────
spec.resources.cpu          (str "4")           →   cpu_request_vcpu   (float 4.0)
spec.resources.memory       (str "8Gi")         →   memory_request_gb  (float 8.0)
spec.resources.gpu          (int  0)            →   gpu_count          (int   0)
spec.resources.storage      (str "50Gi")        →   storage_gb         (float 50.0)
spec.resources.networkBand… (float 1.0)         →   network_bandwidth_gbps (float)
spec.expectedDurationHours  (float 1.0)         →   expected_duration_hours (float)
spec.priority               (int 2)             →   priority           (int 2)
spec.sla.maxLatencyMs       (int 200)           →   sla_latency_ms     (int 200)
spec.workloadType           (str "training")    →   workload_type      (str)
spec.spotTolerant           (bool False)        →   is_spot_tolerant   (int 0/1)
metadata.name               (str)               →   workload_id        (str)
metadata.namespace          (str)               →   namespace          (str)
"""

import logging
import re
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# ── Workload type encoding — must match StateBuilder ─────────────────────────
_WORKLOAD_TYPE_MAP: Dict[str, int] = {
    "training":   0,
    "inference":  1,
    "batch":      2,
    "streaming":  3,
}

# ── Memory unit conversions to GB ─────────────────────────────────────────────
_MEMORY_UNITS: Dict[str, float] = {
    "gi": 1.0,          # GiB ≈ GB for our purposes
    "gb": 1.0,
    "mi": 1.0 / 1024,
    "mb": 1.0 / 1024,
    "ti": 1024.0,
    "tb": 1024.0,
    "ki": 1.0 / (1024 * 1024),
    "kb": 1.0 / (1024 * 1024),
}


def _parse_memory(value: Any, default: float = 4.0) -> float:
    """
    Parses Kubernetes memory strings to GB.
    Examples: "8Gi" → 8.0, "512Mi" → 0.5, "16GB" → 16.0, "4" → 4.0
    """
    if value is None:
        return default
    s = str(value).strip()
    m = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z]*)$", s)
    if not m:
        logger.warning("workload_mapper: cannot parse memory '%s' — using %.1f GB", value, default)
        return default
    num  = float(m.group(1))
    unit = m.group(2).lower()
    mult = _MEMORY_UNITS.get(unit, 1.0)
    return round(num * mult, 3)


def _parse_cpu(value: Any, default: float = 1.0) -> float:
    """
    Parses Kubernetes CPU strings to float vCPUs.
    Examples: "4" → 4.0, "500m" → 0.5, "2.5" → 2.5
    """
    if value is None:
        return default
    s = str(value).strip()
    if s.endswith("m"):
        try:
            return round(float(s[:-1]) / 1000.0, 3)
        except ValueError:
            return default
    try:
        return round(float(s), 3)
    except ValueError:
        logger.warning("workload_mapper: cannot parse cpu '%s' — using %.1f", value, default)
        return default


def _parse_storage(value: Any, default: float = 50.0) -> float:
    """
    Parses storage strings to GB.
    Examples: "100Gi" → 100.0, "500GB" → 500.0
    """
    if value is None:
        return default
    return _parse_memory(value, default)


class WorkloadMapper:
    """
    Converts a CloudWorkload CR dict (from kubectl get -o json)
    into the workload dict format expected by SchedulerAgent.decide().

    Stateless — safe to call from multiple threads.
    """

    def map(self, cr: Dict) -> Optional[Dict]:
        """
        Args:
            cr: Full CloudWorkload CR dict (kubernetes API object)

        Returns:
            workload dict ready for SchedulerAgent.decide(), or None on error.

        Example return:
            {
              "workload_id":             "my-ml-job",
              "namespace":               "velox",
              "cpu_request_vcpu":        4.0,
              "memory_request_gb":       8.0,
              "gpu_count":               0,
              "storage_gb":              50.0,
              "network_bandwidth_gbps":  1.0,
              "expected_duration_hours": 2.0,
              "priority":                2,
              "sla_latency_ms":          200,
              "workload_type":           "training",
              "workload_type_encoded":   0,
              "is_spot_tolerant":        0,
              "constraints":             {...},
            }
        """
        try:
            meta = cr.get("metadata", {})
            spec = cr.get("spec",     {})

            workload_id = meta.get("name",      "unknown")
            namespace   = meta.get("namespace", "velox")
            resources   = spec.get("resources", {})
            sla         = spec.get("sla",        {})
            constraints = spec.get("constraints", {})

            workload_type = spec.get("workloadType", "batch")
            type_encoded  = _WORKLOAD_TYPE_MAP.get(workload_type, 2)

            spot_tolerant = spec.get("spotTolerant", False)
            if isinstance(spot_tolerant, str):
                spot_tolerant = spot_tolerant.lower() in ("true", "1", "yes")

            return {
                "workload_id":             workload_id,
                "namespace":               namespace,
                "cpu_request_vcpu":        _parse_cpu(resources.get("cpu"),     1.0),
                "memory_request_gb":       _parse_memory(resources.get("memory"), 4.0),
                "gpu_count":               int(resources.get("gpu",     0)),
                "storage_gb":              _parse_storage(resources.get("storage", "50Gi")),
                "network_bandwidth_gbps":  float(resources.get("networkBandwidthGbps", 1.0)),
                "expected_duration_hours": float(spec.get("expectedDurationHours", 1.0)),
                "priority":                int(spec.get("priority", 2)),
                "sla_latency_ms":          int(sla.get("maxLatencyMs", 200)),
                "sla_tier":                sla.get("tier", "standard"),
                "workload_type":           workload_type,
                "workload_type_encoded":   type_encoded,
                "is_spot_tolerant":        int(bool(spot_tolerant)),
                "constraints":             constraints,
            }

        except (KeyError, ValueError, TypeError) as exc:
            logger.error("WorkloadMapper.map failed for CR %s: %s",
                         cr.get("metadata", {}).get("name", "?"), exc)
            return None

    def map_list(self, cr_list: Dict) -> list:
        """Maps a kubectl get cloudworkloads -o json response (a List object)."""
        items = cr_list.get("items", [])
        mapped = []
        for item in items:
            result = self.map(item)
            if result:
                mapped.append(result)
        return mapped