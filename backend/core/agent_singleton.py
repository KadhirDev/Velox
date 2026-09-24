"""
Agent Singleton
================
Loads SchedulerAgent and KafkaProducer once at API startup.

FastAPI dependency injection provides them to route handlers.
Loading is done in a background thread during startup.

SHAP is loaded in a second background thread after the PPO model
is ready, so inference is available immediately even if SHAP
takes longer or fails.

This version also performs a small model warmup before marking
the service ready, reducing first-request cold-start latency.
"""

import logging
import os
import threading
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

_agent = None
_producer = None
_lock = threading.Lock()
_ready = False
_init_started = False


def _load_config() -> dict:
    """
    Load config from config/settings.yaml, then override selected values
    using environment variables injected by Kubernetes ConfigMap.
    """
    p = Path("config/settings.yaml")
    cfg = {}

    if p.exists():
        with open(p, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    else:
        logger.warning("config/settings.yaml not found — using env/default config only")

    model_path = os.environ.get("CLOUDOS_MODEL_PATH", "").strip()
    vecnorm_path = os.environ.get("CLOUDOS_VECNORM_PATH", "").strip()
    kafka_boot = os.environ.get("CLOUDOS_KAFKA_BOOTSTRAP", "").strip()

    if model_path:
        cfg.setdefault("model", {})
        cfg["model"]["path"] = model_path

    if vecnorm_path:
        cfg.setdefault("model", {})
        cfg["model"]["vecnorm"] = vecnorm_path

    if kafka_boot:
        cfg.setdefault("kafka", {})
        cfg["kafka"]["bootstrap_servers"] = kafka_boot

    logger.info(
        "Config resolved: model_path=%s vecnorm_path=%s kafka_bootstrap=%s",
        cfg.get("model", {}).get("path", "NOT SET"),
        cfg.get("model", {}).get("vecnorm", "NOT SET"),
        cfg.get("kafka", {}).get("bootstrap_servers", "NOT SET"),
    )

    return cfg


def _warmup_model(agent) -> None:
    """
    Run a few dummy inferences to reduce first-request cold-start latency.

    This is intentionally best-effort and never fatal.
    It runs before _ready=True so readiness only flips after warmup completes.
    """
    if agent is None:
        logger.warning("AgentSingleton: warmup skipped — agent unavailable")
        return

    model = getattr(agent, "_model", None)
    if model is None:
        logger.warning("AgentSingleton: warmup skipped — agent model unavailable")
        return

    try:
        import numpy as np
        import time as _time

        dummy_obs = np.zeros((1, 45), dtype=np.float32)
        t0 = _time.perf_counter()

        # A few passes help amortize initial backend/JIT/runtime setup.
        for _ in range(3):
            model.predict(dummy_obs, deterministic=True)

        elapsed_ms = (_time.perf_counter() - t0) * 1000.0
        logger.info(
            "AgentSingleton: model warmup complete — 3 inferences in %.0fms "
            "(avg ~%.0fms each)",
            elapsed_ms,
            elapsed_ms / 3.0,
        )
    except Exception as exc:
        logger.warning(
            "AgentSingleton: warmup failed (%s) — first real request may be slower",
            exc,
        )


def _initialise() -> None:
    """
    Load the SchedulerAgent and Kafka producer once during API startup.

    Step 1: Load PPO agent without SHAP first
    Step 2: Warm up the model before marking service ready
    Step 3: Load Kafka producer (non-fatal if unavailable)
    Step 4: Mark ready
    Step 5: Load SHAP asynchronously in background
    """
    global _agent, _producer, _ready

    config = _load_config()

    model_path = config.get("model", {}).get("path", "")
    vecnorm_path = config.get("model", {}).get("vecnorm", "")

    mp = Path(model_path) if model_path else Path("")
    if model_path and mp.suffix == "":
        mp = Path(f"{model_path}.zip")

    logger.info(
        "AgentSingleton: resolved model file = %s exists=%s",
        str(mp) if model_path else "NOT SET",
        mp.exists() if model_path else False,
    )
    logger.info(
        "AgentSingleton: vecnorm file = %s exists=%s",
        vecnorm_path if vecnorm_path else "NOT SET",
        Path(vecnorm_path).exists() if vecnorm_path else False,
    )

    # -------------------------------------------------------------------
    # Step 1: Load PPO model without SHAP first
    # -------------------------------------------------------------------
    try:
        from ai_engine.inference.scheduler_agent import SchedulerAgent

        agent = SchedulerAgent.load(
            config=config,
            model_path=str(mp) if model_path else None,
            vecnorm_path=vecnorm_path or None,
            with_explainer=False,
        )

        with _lock:
            _agent = agent

        if agent:
            logger.info("AgentSingleton: SchedulerAgent ready (PPO, SHAP pending)")
        else:
            logger.warning(
                "AgentSingleton: SchedulerAgent.load returned None — heuristic mode"
            )

    except Exception as exc:
        logger.error("AgentSingleton: agent load failed: %s", exc, exc_info=True)

    # -------------------------------------------------------------------
    # Step 2: Warmup model before readiness flips true
    # -------------------------------------------------------------------
    with _lock:
        current_agent = _agent
    _warmup_model(current_agent)

    # -------------------------------------------------------------------
    # Step 3: Kafka producer
    # -------------------------------------------------------------------
    try:
        from ai_engine.kafka.producer import VeloxProducer

        producer = VeloxProducer(config)

        with _lock:
            _producer = producer

        logger.info("AgentSingleton: KafkaProducer ready")

    except Exception as exc:
        logger.warning(
            "AgentSingleton: Kafka unavailable (%s) — decisions won't publish",
            exc,
        )

    # -------------------------------------------------------------------
    # Step 4: Mark ready after model warmup + init path complete
    # -------------------------------------------------------------------
    with _lock:
        _ready = True

    logger.info("AgentSingleton: ready (SHAP loading in background)")

    # -------------------------------------------------------------------
    # Step 5: Load SHAP in separate background thread
    # -------------------------------------------------------------------
    def _load_shap_async() -> None:
        global _agent
        try:
            with _lock:
                current_agent = _agent

            if current_agent is None:
                logger.warning("AgentSingleton: SHAP skipped — agent unavailable")
                return

            current_model = getattr(current_agent, "_model", None)
            if current_model is None:
                logger.warning("AgentSingleton: SHAP skipped — agent model unavailable")
                return

            from ai_engine.explainability.shap_explainer import SHAPExplainer
            from ai_engine.explainability.explanation_formatter import (
                ExplanationFormatter,
            )

            # Mark attempt before load so downstream /explain messaging is accurate
            try:
                object.__setattr__(current_agent, "_shap_init_attempted", True)
            except Exception:
                try:
                    current_agent.__dict__["_shap_init_attempted"] = True
                except Exception:
                    logger.debug(
                        "AgentSingleton: unable to set _shap_init_attempted flag directly"
                    )

            explainer = SHAPExplainer.load(
                model=current_model,
                config=config,
                nsamples=50,
            )

            if explainer is not None:
                formatter = ExplanationFormatter()
                with _lock:
                    if _agent is not None:
                        _agent._explainer = explainer
                        _agent._formatter = formatter

                logger.info("AgentSingleton: SHAP explainer attached (shap_ready=True)")
            else:
                logger.warning(
                    "AgentSingleton: SHAPExplainer.load returned None — "
                    "SHAP disabled. Check background dataset and model."
                )

        except Exception as exc:
            logger.warning(
                "AgentSingleton: SHAP background load failed (%s) — "
                "inference continues without explainability.",
                exc,
            )

    shap_thread = threading.Thread(
        target=_load_shap_async,
        daemon=True,
        name="shap-init",
    )
    shap_thread.start()

    logger.info("AgentSingleton: initialisation complete")


def startup_initialise() -> None:
    """
    Called from FastAPI lifespan on startup — runs in background thread.
    Guarded so repeated calls do not start multiple init threads.
    """
    global _init_started

    with _lock:
        if _init_started:
            logger.info("AgentSingleton: startup_initialise already called — skipping")
            return
        _init_started = True

    t = threading.Thread(target=_initialise, daemon=True, name="agent-init")
    t.start()


def get_agent() -> Optional[object]:
    """
    FastAPI dependency — returns agent or None if still loading / failed.
    """
    with _lock:
        return _agent


def get_producer() -> Optional[object]:
    """
    FastAPI dependency — returns producer or None.
    """
    with _lock:
        return _producer


def is_ready() -> bool:
    """
    Returns True once initialisation path has completed and warmup is done.
    """
    with _lock:
        return _ready