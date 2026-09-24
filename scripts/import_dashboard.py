"""
Grafana Dashboard Importer — Updated for Module E
===================================================
Now correctly points the Grafana datasource to the Prometheus SERVER
at :9091 (not the bridge exporter at :9090).

Changes from Module B version:
  - --prom-port default changed from 9090 to 9091
  - Datasource URL set to http://localhost:9091
  - Warns if trying to point datasource to :9090 (bridge, not Prometheus)

Usage:
  python scripts/import_dashboard.py --grafana-password cloudos123
  python scripts/import_dashboard.py --grafana-password cloudos123 --prom-port 9091
"""

import argparse
import base64
import json
import sys
import time
from typing import Optional
from pathlib import Path

try:
    import requests
except ImportError:
    raise ImportError("pip install requests")

_DASHBOARD_PATH  = Path("infrastructure/grafana/velox_dashboard.json")
_DATASOURCE_PATH = Path("infrastructure/grafana/prometheus_datasource.json")


def _url(host: str, port: int, path: str) -> str:
    return f"http://{host}:{port}{path}"


def _hdrs(user: str, password: str) -> dict:
    creds = base64.b64encode(f"{user}:{password}".encode()).decode()
    return {"Authorization": f"Basic {creds}", "Content-Type": "application/json"}


def wait_for_grafana(host: str, port: int, timeout: int = 30) -> bool:
    url = _url(host, port, "/api/health")
    print(f"[grafana]  Waiting for Grafana at {url} ...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                print(f"[grafana]  ✅ Up — version: {r.json().get('version', '?')}")
                return True
        except requests.ConnectionError:
            pass
        time.sleep(2)
    print(f"[grafana]  ❌ Not reachable after {timeout}s")
    return False


def upsert_datasource(
    grafana_host: str,
    grafana_port: int,
    prom_host:    str,
    prom_port:    int,
    user:         str,
    password:     str,
) -> Optional[str]:
    """
    Creates or updates the Prometheus-CloudOS datasource.
    ALWAYS points to the Prometheus SERVER (prom_port, default 9091),
    NOT the bridge exporter (:9090).
    Returns the datasource UID.
    """
    hdrs     = _hdrs(user, password)
    prom_url = f"http://{prom_host}:{prom_port}"

    if prom_port == 9090:
        print(f"[datasource]  ⚠️  WARNING: prom_port=9090 points to the bridge exporter.")
        print(f"[datasource]             PromQL queries will fail from Grafana.")
        print(f"[datasource]             Use --prom-port 9091 for the Prometheus server.")

    ds_payload = {
        "name":      "Prometheus-CloudOS",
        "type":      "prometheus",
        "url":        prom_url,
        "access":    "proxy",
        "isDefault": True,
        "basicAuth": False,
        "jsonData": {
            "httpMethod":       "POST",
            "prometheusType":   "Prometheus",
            "timeInterval":     "10s",
        },
    }

    # Check if datasource already exists
    r = requests.get(
        _url(grafana_host, grafana_port, "/api/datasources/name/Prometheus-CloudOS"),
        headers=hdrs, timeout=10,
    )

    if r.status_code == 200:
        existing    = r.json()
        existing_id = existing.get("id")
        current_url = existing.get("url", "")

        if current_url == prom_url:
            uid = existing.get("uid", "")
            print(f"[datasource]  ✅ Already configured correctly → {prom_url}  uid={uid}")
            return uid

        # Update to point to correct Prometheus URL
        print(f"[datasource]  Updating datasource URL: {current_url} → {prom_url}")
        r2 = requests.put(
            _url(grafana_host, grafana_port, f"/api/datasources/{existing_id}"),
            headers=hdrs,
            json=ds_payload,
            timeout=10,
        )
        if r2.status_code in (200, 201):
            uid = r2.json().get("datasource", {}).get("uid", "")
            print(f"[datasource]  ✅ Updated → {prom_url}  uid={uid}")
            return uid
        else:
            print(f"[datasource]  ❌ Update failed: {r2.status_code} {r2.text}")
            return None

    # Create new datasource
    r = requests.post(
        _url(grafana_host, grafana_port, "/api/datasources"),
        headers=hdrs,
        json=ds_payload,
        timeout=10,
    )
    if r.status_code in (200, 201):
        uid = r.json().get("datasource", {}).get("uid", "") or r.json().get("uid", "")
        print(f"[datasource]  ✅ Created → {prom_url}  uid={uid}")
        return uid
    else:
        print(f"[datasource]  ❌ Create failed: {r.status_code} {r.text}")
        return None


def import_dashboard(
    grafana_host: str,
    grafana_port: int,
    ds_uid:       str,
    user:         str,
    password:     str,
) -> bool:
    hdrs = _hdrs(user, password)

    with open(_DASHBOARD_PATH, encoding="utf-8") as fh:
        dashboard = json.load(fh)

    # Remove __inputs — substitute datasource UID directly
    dashboard.pop("__inputs",   None)
    dashboard.pop("__requires", None)
    dashboard["id"] = None

    # Replace all datasource references with the actual UID
    dash_str = json.dumps(dashboard)
    dash_str = dash_str.replace("${DS_PROMETHEUS}", ds_uid)
    dash_str = dash_str.replace("DS_PROMETHEUS",    ds_uid)
    dashboard = json.loads(dash_str)

    payload = {
        "dashboard": dashboard,
        "folderId":  0,
        "overwrite": True,
        "message":   "CloudOS-RL Module E import",
    }

    r = requests.post(
        _url(grafana_host, grafana_port, "/api/dashboards/import"),
        headers=hdrs,
        json=payload,
        timeout=15,
    )

    if r.status_code == 200:
        result = r.json()
        url    = result.get("importedUrl", "/d/cloudos-rl-v1")
        print(f"[dashboard]  ✅ Imported successfully!")
        print(f"[dashboard]  🔗 http://{grafana_host}:{grafana_port}{url}")
        return True
    else:
        print(f"[dashboard]  ❌ Import failed: {r.status_code} — {r.text}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grafana-host",    default="localhost")
    parser.add_argument("--grafana-port",    default=3000,  type=int)
    parser.add_argument("--grafana-user",    default="admin")
    parser.add_argument("--grafana-password",default="cloudos123")
    parser.add_argument("--prom-host",       default="localhost")
    parser.add_argument("--prom-port",       default=9091, type=int,
                        help="Prometheus SERVER port (default: 9091). "
                             "Do NOT use 9090 — that is the bridge exporter.")
    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║     CloudOS-RL  Grafana Dashboard Importer       ║")
    print("╚══════════════════════════════════════════════════╝")
    print()
    print(f"  Grafana  : http://{args.grafana_host}:{args.grafana_port}")
    print(f"  Prometheus datasource will point to: http://{args.prom_host}:{args.prom_port}")
    if args.prom_port == 9090:
        print(f"  ⚠️  Port 9090 = bridge exporter (not Prometheus server)")
        print(f"     Use --prom-port 9091 after running: docker compose up -d prometheus")
    print()

    if not _DASHBOARD_PATH.exists():
        print(f"[error]  {_DASHBOARD_PATH} not found.")
        sys.exit(1)

    if not wait_for_grafana(args.grafana_host, args.grafana_port):
        sys.exit(1)

    print()
    ds_uid = upsert_datasource(
        args.grafana_host, args.grafana_port,
        args.prom_host,    args.prom_port,
        args.grafana_user, args.grafana_password,
    )
    if not ds_uid:
        print("[error]  Datasource setup failed.")
        sys.exit(1)

    print()
    ok = import_dashboard(
        args.grafana_host, args.grafana_port,
        ds_uid,
        args.grafana_user, args.grafana_password,
    )
    print()

    if ok:
        print("=" * 54)
        print("  Setup complete")
        print(f"  Dashboard : http://{args.grafana_host}:{args.grafana_port}/d/cloudos-rl-v1")
        print(f"  Datasource: http://{args.prom_host}:{args.prom_port} (Prometheus)")
        print("=" * 54)
    else:
        sys.exit(1)


if __name__ == "__main__":
    # Fix missing Optional import
    from typing import Optional
    main()