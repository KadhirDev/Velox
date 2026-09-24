"""
CloudOS-RL — Module B/E Verification Script (FIXED)
=====================================================
Correctly distinguishes between:

  localhost:9090  → Bridge/Exporter endpoint  (/metrics text format only)
                    NOT a Prometheus query API
                    Cannot run PromQL queries here

  localhost:9091  → Prometheus Server          (/api/v1/query — full PromQL)
                    Started by Module E docker compose
                    This is where PromQL queries must be sent

  localhost:3000  → Grafana                    (/api/health, /api/dashboards)

Usage:
  python scripts/verify_grafana.py
  python scripts/verify_grafana.py --grafana-password cloudos123
  python scripts/verify_grafana.py --prom-port 9091 --grafana-password cloudos123
"""

import argparse
import base64
import json
import sys
import time
from typing import Dict, List, Optional, Tuple

try:
    import requests
except ImportError:
    raise ImportError("pip install requests")

# PromQL queries to verify — sent to Prometheus query API (port 9091)
_PANEL_QUERIES: List[Tuple[str, str]] = [
    ("Panel 1 — Decisions/min",
     "sum(rate(velox_decisions_total[1m])) * 60"),
    ("Panel 2 — Cost savings p50",
     "histogram_quantile(0.50, sum(rate(velox_cost_savings_ratio_bucket[5m])) by (le))"),
    ("Panel 3 — Carbon savings p50",
     "histogram_quantile(0.50, sum(rate(velox_carbon_savings_ratio_bucket[5m])) by (le))"),
    ("Panel 4 — Latency p95",
     "histogram_quantile(0.95, sum(rate(velox_inference_latency_seconds_bucket[2m])) by (le, cloud))"),
    ("Panel 5 — Carbon intensity",
     "velox_carbon_intensity_gco2_per_kwh"),
    ("Panel 6 — RL reward p50",
     "histogram_quantile(0.50, sum(rate(velox_rl_reward_bucket[5m])) by (le))"),
    ("System  — Bridge up",
     "velox_bridge_up"),
    ("System  — Pipeline pricing",
     "velox_pipeline_pricing_fetches_total"),
    ("System  — Pipeline carbon",
     "velox_pipeline_carbon_fetches_total"),
]


# ---------------------------------------------------------------------------
# Section 1: Bridge / Exporter check (port 9090)
# ---------------------------------------------------------------------------
def check_bridge_exporter(host: str, port: int) -> bool:
    """
    Checks the Kafka-Prometheus bridge /metrics endpoint.
    This is a plain text metrics exporter — NOT a Prometheus query API.
    Only checks /metrics is reachable and contains velox_ metrics.
    """
    url = f"http://{host}:{port}/metrics"
    print(f"\n[bridge]  Checking exporter at {url}")
    print(f"[bridge]  NOTE: This is the metrics exporter endpoint (plain text).")
    print(f"[bridge]         PromQL queries must go to Prometheus server (port 9091).")

    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            print(f"[bridge]  ❌ HTTP {r.status_code}")
            return False

        text = r.text
        velox_lines = [l for l in text.split("\n") if l.startswith("velox_") and not l.startswith("# ")]
        metric_names  = set(l.split("{")[0].split(" ")[0] for l in velox_lines)

        print(f"[bridge]  ✅ Reachable — {len(metric_names)} unique velox_* metrics exposed")

        # Show key metrics present
        key_metrics = [
            "velox_bridge_up",
            "velox_carbon_intensity_gco2_per_kwh",
            "velox_pricing_on_demand_usd_per_hr",
            "velox_decisions_total",
        ]
        for m in key_metrics:
            status = "✅" if m in metric_names else "⚠️  (not yet — needs bridge to run longer)"
            print(f"[bridge]    {status}  {m}")

        return True

    except requests.ConnectionError:
        print(f"[bridge]  ❌ Not reachable at {url}")
        print(f"[bridge]     Start bridge: python -m ai_engine.kafka.kafka_prometheus_bridge")
        return False
    except Exception as exc:
        print(f"[bridge]  ❌ Error: {exc}")
        return False


# ---------------------------------------------------------------------------
# Section 2: Prometheus Server check (port 9091)
# ---------------------------------------------------------------------------
def check_prometheus_server(host: str, port: int) -> bool:
    """
    Checks the real Prometheus server started by Module E docker compose.
    Verifies /api/v1/query endpoint works (full PromQL support).
    """
    base_url = f"http://{host}:{port}"
    print(f"\n[prometheus]  Checking Prometheus server at {base_url}")
    print(f"[prometheus]  NOTE: This is the REAL Prometheus server (PromQL query API).")
    print(f"[prometheus]         Started by: docker compose up -d prometheus")

    # Health check
    try:
        r = requests.get(f"{base_url}/-/healthy", timeout=5)
        if r.status_code == 200:
            print(f"[prometheus]  ✅ Healthy")
        else:
            print(f"[prometheus]  ❌ Health check returned {r.status_code}")
            return False
    except requests.ConnectionError:
        print(f"[prometheus]  ❌ Not reachable at {base_url}")
        print(f"[prometheus]     Start it: docker compose up -d prometheus")
        print(f"[prometheus]     Or run natively — see scripts/start_stack.ps1")
        return False

    # Check it is scraping the bridge
    try:
        r = requests.get(f"{base_url}/api/v1/targets", timeout=5)
        if r.status_code == 200:
            targets = r.json().get("data", {}).get("activeTargets", [])
            bridge_targets = [t for t in targets if "cloudos" in t.get("job", "")]
            if bridge_targets:
                state = bridge_targets[0].get("health", "unknown")
                last  = bridge_targets[0].get("lastScrape", "")
                print(f"[prometheus]  ✅ Scraping bridge — health={state} lastScrape={last[:19]}")
            else:
                print(f"[prometheus]  ⚠️  No cloudos-bridge target found yet (may take 15s)")
        else:
            print(f"[prometheus]  ⚠️  Could not check targets: HTTP {r.status_code}")
    except Exception as exc:
        print(f"[prometheus]  ⚠️  Target check error: {exc}")

    return True


# ---------------------------------------------------------------------------
# Section 3: PromQL queries (port 9091 — Prometheus server)
# ---------------------------------------------------------------------------
def check_panel_queries(host: str, port: int) -> bool:
    """
    Runs all 9 dashboard PromQL queries against the Prometheus server.
    These MUST go to the Prometheus server (:9091), NOT the bridge (:9090).
    """
    base_url = f"http://{host}:{port}/api/v1/query"
    print(f"\n[queries]  Running {len(_PANEL_QUERIES)} dashboard PromQL queries against {host}:{port} ...")

    ok_count   = 0
    data_count = 0

    for name, query in _PANEL_QUERIES:
        try:
            r = requests.get(base_url, params={"query": query}, timeout=8)

            if r.status_code != 200:
                print(f"  ❌  {name}  (HTTP {r.status_code})")
                continue

            # Parse JSON response from Prometheus query API
            body    = r.json()
            status  = body.get("status", "")
            results = body.get("data", {}).get("result", [])

            if status != "success":
                print(f"  ❌  {name}  (status={status})")
                continue

            ok_count += 1
            if results:
                data_count += 1
                val = results[0].get("value", [None, ""])[1] if results else ""
                print(f"  ✅  {name}  → {val}")
            else:
                print(f"  ⚠️   {name}  → query OK, no data yet (send test decisions)")

        except requests.ConnectionError:
            print(f"  ❌  {name}  (Prometheus not reachable)")
            return False
        except json.JSONDecodeError as exc:
            # This was the original error — response was not valid JSON
            # Happens when querying the bridge (:9090) instead of Prometheus (:9091)
            print(f"  ❌  {name}  (JSON decode error — are you querying the bridge instead of Prometheus?)")
            print(f"       Expected Prometheus server at :{port}, got non-JSON response.")
            print(f"       Fix: run 'docker compose up -d prometheus' to start Prometheus at :9091")
            return False
        except Exception as exc:
            print(f"  ❌  {name}  ({exc})")

    print(f"\n[queries]  {ok_count}/{len(_PANEL_QUERIES)} queries valid | {data_count} with live data")
    return ok_count == len(_PANEL_QUERIES)


# ---------------------------------------------------------------------------
# Section 4: Grafana check (port 3000)
# ---------------------------------------------------------------------------
def check_grafana(host: str, port: int, user: str, password: str) -> bool:
    """Checks Grafana is running, datasource exists, dashboard exists."""
    creds = base64.b64encode(f"{user}:{password}".encode()).decode()
    hdrs  = {"Authorization": f"Basic {creds}", "Content-Type": "application/json"}
    base  = f"http://{host}:{port}"

    print(f"\n[grafana]  Checking http://{host}:{port} ...")

    try:
        r = requests.get(f"{base}/api/health", timeout=5)
        if r.status_code == 200:
            info = r.json()
            print(f"[grafana]  ✅ Running — version: {info.get('version', 'unknown')}")
        else:
            print(f"[grafana]  ❌ HTTP {r.status_code}")
            return False
    except requests.ConnectionError:
        print(f"[grafana]  ❌ Not reachable at {host}:{port}")
        return False

    # Check datasource
    r = requests.get(f"{base}/api/datasources/name/Prometheus-CloudOS", headers=hdrs, timeout=5)
    if r.status_code == 200:
        ds      = r.json()
        ds_url  = ds.get("url", "")
        ds_uid  = ds.get("uid", "")
        print(f"[grafana]  ✅ Datasource 'Prometheus-CloudOS' exists")
        print(f"[grafana]     URL: {ds_url}  UID: {ds_uid}")

        # Warn if datasource points to bridge (:9090) instead of Prometheus (:9091)
        if ":9090" in ds_url:
            print(f"[grafana]  ⚠️  IMPORTANT: Datasource points to :9090 (bridge exporter).")
            print(f"[grafana]              PromQL queries will fail.")
            print(f"[grafana]              Fix: run python scripts/import_dashboard.py --prom-port 9091")
            print(f"[grafana]              This updates the datasource to point to Prometheus :9091")
        elif ":9091" in ds_url:
            print(f"[grafana]  ✅ Datasource correctly points to Prometheus server :9091")
    else:
        print(f"[grafana]  ⚠️  Datasource not found — run: python scripts/import_dashboard.py")

    # Check dashboard
    r = requests.get(f"{base}/api/dashboards/uid/cloudos-rl-v1", headers=hdrs, timeout=5)
    if r.status_code == 200:
        title = r.json().get("dashboard", {}).get("title", "")
        print(f"[grafana]  ✅ Dashboard '{title}' exists")
        print(f"[grafana]  🔗 http://{host}:{port}/d/cloudos-rl-v1")
    else:
        print(f"[grafana]  ⚠️  Dashboard not found — run: python scripts/import_dashboard.py")

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Verify CloudOS-RL monitoring stack")
    parser.add_argument("--bridge-host",     default="localhost")
    parser.add_argument("--bridge-port",     default=9090, type=int,
                        help="Bridge/exporter port (default: 9090)")
    parser.add_argument("--prom-host",       default="localhost")
    parser.add_argument("--prom-port",       default=9091, type=int,
                        help="Prometheus SERVER port (default: 9091, started by docker compose)")
    parser.add_argument("--grafana-host",    default="localhost")
    parser.add_argument("--grafana-port",    default=3000, type=int)
    parser.add_argument("--grafana-user",    default="admin")
    parser.add_argument("--grafana-password",default="cloudos123")
    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║    CloudOS-RL  Full Stack Verification           ║")
    print("╚══════════════════════════════════════════════════╝")
    print()
    print("  Port map:")
    print(f"    :9090  → Bridge/Exporter    (plain text /metrics)")
    print(f"    :9091  → Prometheus Server  (PromQL query API)")
    print(f"    :3000  → Grafana            (dashboards)")

    bridge_ok   = check_bridge_exporter(args.bridge_host,  args.bridge_port)
    prom_ok     = check_prometheus_server(args.prom_host,  args.prom_port)
    query_ok    = check_panel_queries(args.prom_host, args.prom_port) if prom_ok else False
    grafana_ok  = check_grafana(args.grafana_host, args.grafana_port,
                                args.grafana_user, args.grafana_password)

    print()
    print("=" * 54)
    print("  Summary")
    print("=" * 54)
    print(f"  Bridge exporter  :9090  : {'✅ OK' if bridge_ok  else '❌ Start bridge'}")
    print(f"  Prometheus       :9091  : {'✅ OK' if prom_ok    else '❌ docker compose up -d prometheus'}")
    print(f"  PromQL queries          : {'✅ OK' if query_ok   else '⚠️  No data yet or Prometheus not up'}")
    print(f"  Grafana          :3000  : {'✅ OK' if grafana_ok else '❌ Start Grafana'}")
    print("=" * 54)

    if prom_ok and grafana_ok:
        print()
        print(f"  Prometheus UI : http://localhost:{args.prom_port}")
        print(f"  Dashboard     : http://localhost:{args.grafana_port}/d/cloudos-rl-v1")
        print()

    if not prom_ok:
        print()
        print("  NEXT STEP: Start Prometheus server")
        print("  Run:  docker compose up -d prometheus")
        print("  Then re-run this script.")
        print()


if __name__ == "__main__":
    main()