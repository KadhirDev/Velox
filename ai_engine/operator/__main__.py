"""
Velox Operator entrypoint
==============================
Run with:
  python -m ai_engine.operator
  python -m ai_engine.operator --dry-run
  python -m ai_engine.operator --no-shap --poll-interval 10
  python -m ai_engine.operator --run-once
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml


def _setup_logging(level: str):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(name)-42s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _load_config() -> dict:
    path = Path("config/settings.yaml")
    if not path.exists():
        logging.warning("config/settings.yaml not found — using empty config")
        return {}
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def main():
    parser = argparse.ArgumentParser(
        description="Velox Kubernetes Operator — watches CloudWorkload CRs"
    )
    parser.add_argument("--namespace",     default="velox",
                        help="Kubernetes namespace to watch (default: velox)")
    parser.add_argument("--poll-interval", default=5, type=int,
                        help="Seconds between poll cycles (default: 5)")
    parser.add_argument("--dry-run",       action="store_true",
                        help="Log decisions but do not patch CRs or publish to Kafka")
    parser.add_argument("--no-kafka",      action="store_true",
                        help="Make decisions and patch CRs but skip Kafka publishing")
    parser.add_argument("--no-shap",       action="store_true",
                        help="Skip SHAP explainability (faster, no explanation field)")
    parser.add_argument("--run-once",      action="store_true",
                        help="Run one poll cycle and exit (useful for testing)")
    parser.add_argument("--log-level",     default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    _setup_logging(args.log_level)
    log = logging.getLogger("velox.operator")

    log.info("=" * 60)
    log.info("  Velox Operator")
    log.info("  namespace=%s  dry_run=%s  no_kafka=%s  no_shap=%s",
             args.namespace, args.dry_run, args.no_kafka, args.no_shap)
    log.info("=" * 60)

    config = _load_config()

    from ai_engine.operator.operator import VeloxOperator
    operator = VeloxOperator(
        config=config,
        dry_run=args.dry_run,
        no_kafka=args.no_kafka,
        no_shap=args.no_shap,
        namespace=args.namespace,
        poll_interval=args.poll_interval,
    )

    if args.run_once:
        n = operator.run_once()
        log.info("run-once: processed %d workload(s)", n)
        sys.exit(0)
    else:
        operator.start()


if __name__ == "__main__":
    main()