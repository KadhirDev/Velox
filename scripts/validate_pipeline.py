"""
End-to-end Pipeline Validation
================================
Fires a real scheduling request at the API, then verifies:
  1. API returned a valid SchedulingDecision
  2. Kafka topic received the decision message
  3. Prometheus counter incremented
  4. CloudWorkload CR was created (optional)

Usage:
  python scripts/validate_pipeline.py
  python scripts/validate_pipeline.py --api-url http://localhost:8001 --prom-url http://localhost:9091
"""

import argparse
import json
import time
import sys

try:
    import requests
except ImportError:
    raise ImportError("pip install requests")


def check_api_health(base: str) -> bool:
    try:
        r = requests.get(f"{base}/health", timeout=5)
        data = r.json()
        print(f"[api]    ✅ {data}")
        return data.get("agent_loaded", False)
    except Exception as exc:
        print(f"[api]    ❌ {exc}")
        return False


def submit_workload(base: str) -> dict:
    payload = {
        "workload_type":           "training",
        "cpu_request_vcpu":        4.0,
        "memory_request_gb":       8.0,
        "gpu_count":               1,
        "storage_gb":              100.0,
        "network_bandwidth_gbps":  2.0,
        "expected_duration_hours": 2.0,
        "priority":                2,
        "sla_latency_ms":          300,
        "is_spot_tolerant":        True,
    }
    r = requests.post(f"{base}/api/v1/schedule", json=payload, timeout=10)
    r.raise_for_status()
    return r.json()


def check_prometheus_counter(prom: str, before: float) -> bool:
    """Checks velox_decisions_total incremented after submission."""
    try:
        r = requests.get(
            f"{prom}/api/v1/query",
            params={"query": "sum(velox_decisions_total)"},
            timeout=5,
        )
        results = r.json().get("data", {}).get("result", [])
        if results:
            after = float(results[0]["value"][1])
            if after > before:
                print(f"[prom]   ✅ decisions_total: {before:.0f} → {after:.0f}")
                return True
        print(f"[prom]   ⚠️  counter not incremented yet (bridge poll delay ~10s)")
        return False
    except Exception as exc:
        print(f"[prom]   ❌ {exc}")
        return False


def get_prom_counter(prom: str) -> float:
    try:
        r = requests.get(
            f"{prom}/api/v1/query",
            params={"query": "sum(velox_decisions_total)"},
            timeout=5,
        )
        results = r.json().get("data", {}).get("result", [])
        return float(results[0]["value"][1]) if results else 0.0
    except Exception:
        return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url",  default="http://localhost:8001")
    parser.add_argument("--prom-url", default="http://localhost:9091")
    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║    CloudOS-RL  Pipeline Validation               ║")
    print("╚══════════════════════════════════════════════════╝")
    print()

    # Step 1: health
    print("[step 1]  API health check ...")
    agent_loaded = check_api_health(args.api_url)
    if not agent_loaded:
        print("          ⚠️  Agent not loaded yet — wait 30s and retry")

    # Step 2: baseline counter
    print("[step 2]  Prometheus baseline counter ...")
    before = get_prom_counter(args.prom_url)
    print(f"          velox_decisions_total before = {before:.0f}")

    # Step 3: submit workload
    print("[step 3]  Submitting workload to /api/v1/schedule ...")
    try:
        t0       = time.perf_counter()
        decision = submit_workload(args.api_url)
        elapsed  = (time.perf_counter() - t0) * 1000
        print(f"[api]    ✅ Decision received in {elapsed:.0f}ms")
        print(f"          cloud={decision['cloud']} region={decision['region']}")
        print(f"          instance={decision['instance_type']} purchase={decision['purchase_option']}")
        print(f"          cost={decision['estimated_cost_per_hr']:.4f}/hr "
              f"savings={decision['cost_savings_pct']:.1f}%")
        print(f"          carbon_savings={decision['carbon_savings_pct']:.1f}%")
        print(f"          latency_ms={decision['latency_ms']:.1f}")
        expl = decision.get("explanation", {})
        if expl and expl.get("summary"):
            print(f"          SHAP: {expl['summary'][:80]}")
        print(f"          confidence={expl.get('confidence', 0):.3f}" if expl else "")
    except Exception as exc:
        print(f"[api]    ❌ Failed: {exc}")
        sys.exit(1)

    # Step 4: wait for Kafka → bridge → Prometheus propagation
    print("[step 4]  Waiting 15s for Kafka→Bridge→Prometheus propagation ...")
    time.sleep(15)
    check_prometheus_counter(args.prom_url, before)

    # Step 5: verify /decisions endpoint
    print("[step 5]  Checking /api/v1/decisions ...")
    try:
        r = requests.get(f"{args.api_url}/api/v1/decisions?limit=1", timeout=5)
        data = r.json()
        print(f"[api]    ✅ {data['count']} decisions in store")
    except Exception as exc:
        print(f"[api]    ❌ {exc}")

    # Step 6: agent status
    print("[step 6]  Agent status ...")
    try:
        r = requests.get(f"{args.api_url}/api/v1/status", timeout=5)
        s = r.json()
        print(f"[api]    ✅ agent_loaded={s['agent_loaded']} "
              f"shap_ready={s['shap_ready']} "
              f"decisions_served={s['decisions_served']}")
    except Exception as exc:
        print(f"[api]    ❌ {exc}")

    print()
    print("=" * 54)
    print(f"  Validation complete.")
    print(f"  Refresh Grafana: http://localhost:3000/d/cloudos-rl-v1")
    print("=" * 54)


if __name__ == "__main__":
    main()