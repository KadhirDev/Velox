"""
SchedulerAgent
===============
Loads the trained PPO model and runs inference.
Integrates SHAP explainability into every decision when available.

Optimized safely for concurrent inference:
  - preserves existing public API and backward compatibility
  - keeps robust model / vecnorm / numpy-pickle loading
  - uses lock-free VecNormalize math when stats are available
  - uses stampede-safe carbon TTL cache
  - reduces repeated work in hot path
  - keeps request/response behavior unchanged
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import pickle
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_MODEL_PATH = Path("models/best/best_model")
_VECNORM_PATH = Path("models/vec_normalize.pkl")

# -------------------------------------------------------------------------
# Module-level carbon TTL cache with stampede protection
# -------------------------------------------------------------------------
_carbon_cache_lock = threading.Lock()
_carbon_cache_data: Dict[str, float] = {}
_carbon_cache_ts: float = 0.0
_carbon_cache_refreshing: bool = False
_CARBON_CACHE_TTL = 60.0  # seconds


class SchedulerAgent:
    """
    Inference wrapper for the trained PPO scheduling agent.
    Loads model once, serves many requests, with optional SHAP explanations.
    """

    # -------------------------------------------------------------------------
    # Static fallback maps for pricing / carbon estimation
    # -------------------------------------------------------------------------
    _STATIC_ON_DEMAND: Dict[str, float] = {
        # AWS
        "us-east-1": 0.096,
        "us-east-2": 0.096,
        "us-west-1": 0.115,
        "us-west-2": 0.096,
        "eu-west-1": 0.107,
        "eu-west-2": 0.113,
        "eu-west-3": 0.113,
        "eu-central-1": 0.111,
        "eu-north-1": 0.098,
        "ap-southeast-1": 0.114,
        "ap-southeast-2": 0.121,
        "ap-northeast-1": 0.118,
        "ap-northeast-2": 0.114,
        "ap-south-1": 0.115,
        "sa-east-1": 0.139,
        "ca-central-1": 0.100,
        # GCP
        "us-central1": 0.096,
        "europe-west4": 0.107,
        "asia-southeast1": 0.114,
        # Azure
        "eastus": 0.096,
        "eastasia": 0.121,
        "westeurope": 0.107,
        "westus2": 0.096,
    }

    _STATIC_CARBON: Dict[str, float] = {
        # AWS
        "us-east-1": 415.0,
        "us-east-2": 410.0,
        "us-west-1": 252.0,
        "us-west-2": 192.0,
        "eu-west-1": 316.0,
        "eu-west-2": 225.0,
        "eu-west-3": 58.0,
        "eu-central-1": 338.0,
        "eu-north-1": 42.0,
        "ap-southeast-1": 453.0,
        "ap-southeast-2": 610.0,
        "ap-northeast-1": 506.0,
        "ap-northeast-2": 415.0,
        "ap-south-1": 708.0,
        "sa-east-1": 136.0,
        "ca-central-1": 89.0,
        # GCP
        "us-central1": 360.0,
        "europe-west4": 284.0,
        "asia-southeast1": 453.0,
        # Azure
        "eastus": 400.0,
        "eastasia": 506.0,
        "westeurope": 290.0,
        "westus2": 220.0,
    }

    _SPOT_DISCOUNT: float = 0.65
    _RESERVED_1YR: float = 0.60
    _RESERVED_3YR: float = 0.40
    _BASELINE_REGION: str = "us-east-1"

    def __init__(
        self,
        model: Any,
        vec_env: Any,
        config: Dict[str, Any],
        explainer: Any = None,
        formatter: Any = None,
    ) -> None:
        self._model = model
        self._vec_env = vec_env
        self._config = config or {}
        self._explainer = explainer
        self._formatter = formatter

        from ai_engine.environment.action_decoder import ActionDecoder
        from ai_engine.environment.state_builder import StateBuilder
        from ai_engine.cloud_adapter.pricing_cache import PricingCache

        self._decoder = ActionDecoder()
        self._state_builder = StateBuilder(self._config)
        self._pricing_cache = PricingCache(self._config)

        # Extract VecNormalize statistics once so the request path can avoid
        # vec_env.normalize_obs() lock contention when available.
        self._obs_mean: Optional[np.ndarray] = None
        self._obs_var: Optional[np.ndarray] = None
        self._obs_clip: float = 10.0
        self._has_vecnorm_stats: bool = False
        self._extract_vecnorm_stats()

        # Reduce warning spam under load while keeping visibility.
        self._slow_infer_warn_ms = 800.0
        self._slow_data_warn_ms = 100.0

    # -------------------------------------------------------------------------
    # Factory
    # -------------------------------------------------------------------------
    @classmethod
    def load(
        cls,
        config: Dict[str, Any],
        model_path: Optional[str] = None,
        vecnorm_path: Optional[str] = None,
        with_explainer: bool = True,
        force_bg_regen: bool = False,
    ) -> Optional["SchedulerAgent"]:
        """
        Load PPO model + optional VecNormalize + optional SHAP explainer.

        Path resolution priority:
          1) explicit function args
          2) environment variables
          3) config dict
          4) built-in defaults
        """
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv

        mp_str = (
            model_path
            or os.environ.get("CLOUDOS_MODEL_PATH", "")
            or config.get("model", {}).get("path", str(_MODEL_PATH))
        )
        vp_str = (
            vecnorm_path
            or os.environ.get("CLOUDOS_VECNORM_PATH", "")
            or config.get("model", {}).get("vecnorm", str(_VECNORM_PATH))
        )

        mp = Path(mp_str)
        vp = Path(vp_str)
        model_file = mp if mp.suffix else Path(f"{mp}.zip")

        logger.info(
            "SchedulerAgent.load: model_file=%s exists=%s",
            model_file,
            model_file.exists(),
        )
        logger.info(
            "SchedulerAgent.load: vecnorm=%s exists=%s",
            vp,
            vp.exists(),
        )

        if not model_file.exists():
            logger.error(
                "SchedulerAgent: model file not found: %s\n"
                "  Files in parent dir: %s",
                model_file,
                list(model_file.parent.glob("*")) if model_file.parent.exists() else "dir missing",
            )
            return None

        try:
            cls._install_numpy_pickle_compat()

            custom_objects = {
                "learning_rate": 0.0003,
                "lr_schedule": lambda _: 0.0003,
                "clip_range": lambda _: 0.2,
            }
            model = PPO.load(
                str(model_file),
                device="cpu",
                custom_objects=custom_objects,
            )
            logger.info("SchedulerAgent: PPO loaded from %s", model_file)
        except Exception as exc:
            logger.error("SchedulerAgent: model load failed: %s", exc, exc_info=True)
            return None

        vec_env = None
        if vp.exists():
            try:
                from ai_engine.environment.cloud_env import VeloxEnv

                dummy = DummyVecEnv([lambda: VeloxEnv(config)])

                cls._install_numpy_pickle_compat()

                with open(vp, "rb") as fh:
                    vec_env = pickle.load(fh)

                if hasattr(vec_env, "set_venv"):
                    vec_env.set_venv(dummy)
                    vec_env.training = False
                    vec_env.norm_reward = False
                    logger.info("SchedulerAgent: VecNormalize loaded from %s", vp)
                else:
                    logger.warning(
                        "SchedulerAgent: unpickled vecnorm has no set_venv(); running unnormalised."
                    )
                    vec_env = None

            except Exception as exc:
                logger.warning(
                    "SchedulerAgent: VecNormalize load failed (%s) — running unnormalised.",
                    exc,
                    exc_info=True,
                )
                vec_env = None
        else:
            logger.warning("SchedulerAgent: %s not found — running unnormalised.", vp)

        explainer = None
        formatter = None

        if with_explainer:
            try:
                from ai_engine.explainability.shap_explainer import SHAPExplainer
                from ai_engine.explainability.explanation_formatter import ExplanationFormatter

                if hasattr(SHAPExplainer, "load"):
                    explainer = SHAPExplainer.load(
                        model=model,
                        config=config,
                        nsamples=100,
                        force_regen=force_bg_regen,
                    )
                else:
                    explainer = SHAPExplainer(
                        model=model,
                        config=config,
                        nsamples=100,
                        force_regen=force_bg_regen,
                    )

                formatter = ExplanationFormatter()
                logger.info("SchedulerAgent: SHAP explainer ready")

            except Exception as exc:
                logger.warning(
                    "SchedulerAgent: SHAP init failed (%s) — continuing without explanations.",
                    exc,
                )
                explainer = None
                formatter = None

        return cls(
            model=model,
            vec_env=vec_env,
            config=config,
            explainer=explainer,
            formatter=formatter,
        )

    @staticmethod
    def _install_numpy_pickle_compat() -> None:
        """
        Install compatibility aliases for NumPy pickle module paths.
        """
        try:
            core_mod = importlib.import_module("numpy.core")
        except Exception:
            core_mod = None

        try:
            underscore_core_mod = importlib.import_module("numpy._core")
        except Exception:
            underscore_core_mod = None

        if core_mod is not None and underscore_core_mod is None:
            sys.modules.setdefault("numpy._core", core_mod)
            underscore_core_mod = core_mod

        if underscore_core_mod is not None and core_mod is None:
            sys.modules.setdefault("numpy.core", underscore_core_mod)
            core_mod = underscore_core_mod

        submodule_pairs = [
            ("numpy.core.numeric", "numpy._core.numeric"),
            ("numpy.core.multiarray", "numpy._core.multiarray"),
            ("numpy.core.umath", "numpy._core.umath"),
            ("numpy.core._multiarray_umath", "numpy._core._multiarray_umath"),
        ]

        for public_name, private_name in submodule_pairs:
            public_mod = None
            private_mod = None

            try:
                public_mod = importlib.import_module(public_name)
            except Exception:
                public_mod = None

            try:
                private_mod = importlib.import_module(private_name)
            except Exception:
                private_mod = None

            if public_mod is not None and private_mod is None:
                sys.modules.setdefault(private_name, public_mod)

            if private_mod is not None and public_mod is None:
                sys.modules.setdefault(public_name, private_mod)

    # -------------------------------------------------------------------------
    # Public status helpers
    # -------------------------------------------------------------------------
    @property
    def model(self) -> Any:
        return self._model

    @property
    def vec_env(self) -> Any:
        return self._vec_env

    @property
    def explainer(self) -> Any:
        return self._explainer

    @property
    def formatter(self) -> Any:
        return self._formatter

    def is_model_ready(self) -> bool:
        return self._model is not None

    def has_explainer(self) -> bool:
        return self._explainer is not None

    # -------------------------------------------------------------------------
    # VecNormalize optimization
    # -------------------------------------------------------------------------
    def _extract_vecnorm_stats(self) -> None:
        """
        Extract running mean/var from VecNormalize once at startup.
        Under concurrency this avoids per-request lock contention inside
        vec_env.normalize_obs() when stats are fixed for inference.
        """
        if self._vec_env is None:
            return

        try:
            obs_rms = getattr(self._vec_env, "obs_rms", None)
            if obs_rms is None:
                return

            mean = getattr(obs_rms, "mean", None)
            var = getattr(obs_rms, "var", None)
            if mean is None or var is None:
                return

            self._obs_mean = np.asarray(mean, dtype=np.float32).copy()
            self._obs_var = np.asarray(var, dtype=np.float32).copy()
            self._obs_clip = float(getattr(self._vec_env, "clip_obs", 10.0))
            self._has_vecnorm_stats = True

            logger.info(
                "SchedulerAgent: extracted VecNormalize stats for lock-free normalization; shape=%s",
                self._obs_mean.shape,
            )
        except Exception as exc:
            logger.warning(
                "SchedulerAgent: could not extract VecNormalize stats (%s) — falling back to vec_env.normalize_obs().",
                exc,
            )
            self._obs_mean = None
            self._obs_var = None
            self._has_vecnorm_stats = False

    # -------------------------------------------------------------------------
    # Cost / carbon helper methods
    # -------------------------------------------------------------------------
    def _get_pricing(self) -> Dict[str, Any]:
        pricing: Dict[str, Any] = {}

        if self._pricing_cache is None:
            return pricing

        for method_name in ("get_current_pricing", "get_pricing"):
            method = getattr(self._pricing_cache, method_name, None)
            if callable(method):
                try:
                    pricing = method() or {}
                    break
                except Exception as exc:
                    logger.debug("_get_pricing: %s failed (%s)", method_name, exc)

        return pricing

    def _od_price_for_region(self, region: str, pricing: Optional[Dict[str, Any]] = None) -> float:
        """Returns on-demand $/hr, trying PricingCache first then static table."""
        region = str(region or self._BASELINE_REGION)
        flat = pricing if pricing is not None else self._get_pricing()

        try:
            if flat:
                if region in flat:
                    value = flat[region]
                    if isinstance(value, (int, float)):
                        price = float(value)
                    elif isinstance(value, dict):
                        price = float(value.get("on_demand_per_vcpu_hr", 0.0))
                    else:
                        price = 0.0
                    if price > 0:
                        return price

                baseline = flat.get(self._BASELINE_REGION)
                if baseline is not None:
                    if isinstance(baseline, (int, float)):
                        price = float(baseline)
                    elif isinstance(baseline, dict):
                        price = float(baseline.get("on_demand_per_vcpu_hr", 0.0))
                    else:
                        price = 0.0
                    if price > 0:
                        return price
        except Exception:
            pass

        return self._STATIC_ON_DEMAND.get(region, self._STATIC_ON_DEMAND[self._BASELINE_REGION])

    def _carbon_for_region(self, region: str, carbon: Optional[Dict[str, float]] = None) -> float:
        """Returns gCO2/kWh, trying pipeline file first then static table."""
        region = str(region or self._BASELINE_REGION)
        data = carbon if carbon is not None else self._load_carbon()

        try:
            if data:
                if region in data:
                    val = data[region]
                    parsed = float(val.get("gco2_per_kwh", val) if isinstance(val, dict) else val)
                    if parsed > 0:
                        return parsed

                if self._BASELINE_REGION in data:
                    val = data[self._BASELINE_REGION]
                    parsed = float(val.get("gco2_per_kwh", val) if isinstance(val, dict) else val)
                    if parsed > 0:
                        return parsed
        except Exception:
            pass

        return self._STATIC_CARBON.get(region, self._STATIC_CARBON[self._BASELINE_REGION])

    def estimate_cost_per_hr(self, decoded: Dict[str, Any]) -> float:
        region = str(decoded.get("region", self._BASELINE_REGION))
        purchase = str(decoded.get("purchase_option", "on_demand")).lower()
        od = self._od_price_for_region(region)

        if purchase in {"spot", "preemptible"}:
            return round(od * (1.0 - self._SPOT_DISCOUNT), 6)
        if purchase == "reserved_1yr":
            return round(od * self._RESERVED_1YR, 6)
        if purchase == "reserved_3yr":
            return round(od * self._RESERVED_3YR, 6)

        return round(od, 6)

    def cost_savings_pct(self, decoded: Dict[str, Any]) -> float:
        baseline = self._od_price_for_region(self._BASELINE_REGION)
        actual = self.estimate_cost_per_hr(decoded)

        if baseline <= 0:
            return 0.0

        return round(max(0.0, (1.0 - actual / baseline) * 100.0), 2)

    def carbon_savings_pct(self, decoded: Dict[str, Any]) -> float:
        region = str(decoded.get("region", self._BASELINE_REGION))
        actual = self._carbon_for_region(region)
        baseline = self._carbon_for_region(self._BASELINE_REGION)

        if baseline <= 0:
            return 0.0

        return round(max(0.0, (1.0 - actual / baseline) * 100.0), 2)

    def _estimate_cost_cached(self, decoded: Dict[str, Any], pricing: Dict[str, Any]) -> float:
        region = str(decoded.get("region", self._BASELINE_REGION))
        purchase = str(decoded.get("purchase_option", "on_demand")).lower()
        od = self._od_price_for_region(region, pricing)

        if purchase in {"spot", "preemptible"}:
            return round(od * (1.0 - self._SPOT_DISCOUNT), 6)
        if purchase == "reserved_1yr":
            return round(od * self._RESERVED_1YR, 6)
        if purchase == "reserved_3yr":
            return round(od * self._RESERVED_3YR, 6)

        return round(od, 6)

    def _cost_savings_cached(self, decoded: Dict[str, Any], pricing: Dict[str, Any]) -> float:
        baseline = self._od_price_for_region(self._BASELINE_REGION, pricing)
        actual = self._estimate_cost_cached(decoded, pricing)
        if baseline <= 0:
            return 0.0
        return round(max(0.0, (1.0 - actual / baseline) * 100.0), 2)

    def _carbon_savings_cached(self, decoded: Dict[str, Any], carbon: Dict[str, float]) -> float:
        region = str(decoded.get("region", self._BASELINE_REGION))
        actual = self._carbon_for_region(region, carbon)
        baseline = self._carbon_for_region(self._BASELINE_REGION, carbon)
        if baseline <= 0:
            return 0.0
        return round(max(0.0, (1.0 - actual / baseline) * 100.0), 2)

    # -------------------------------------------------------------------------
    # Core inference
    # -------------------------------------------------------------------------
    def schedule(
        self,
        workload: Any,
        include_explanation: bool = True,
    ) -> Dict[str, Any]:
        """
        Main inference entrypoint.
        Delegates to decide() so timing/caching improvements are shared across
        all call sites.
        """
        return self.decide(workload=workload, include_explanation=include_explanation)

    def decide(
        self,
        workload: Any,
        include_explanation: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Main inference entry point.
        Keeps API behavior unchanged while reducing repeated work and contention.

        torch.inference_mode() reduces per-call overhead during PPO prediction:
          - disables autograd tracking
          - disables view tracking
          - safe because this path is inference-only
        """
        if self._model is None:
            logger.error("SchedulerAgent.decide: model not loaded.")
            return None

        t_total = time.perf_counter()

        workload_dict = self._to_dict(workload)
        if "cpu_request_vcpu" in workload_dict and "cpu_request" not in workload_dict:
            workload_dict["cpu_request"] = workload_dict["cpu_request_vcpu"]

        # Step 1: load pricing + carbon
        t0 = time.perf_counter()
        pricing = self._get_pricing()
        carbon = self._load_carbon()
        t_data = (time.perf_counter() - t0) * 1000.0

        # Step 2: build state
        t0 = time.perf_counter()
        state = self._build_state_with(workload_dict, pricing, carbon)
        state = self._ensure_state_array(state)
        t_state = (time.perf_counter() - t0) * 1000.0

        # Step 3: normalise observation
        t0 = time.perf_counter()
        norm_state = self._normalise_obs(state)
        t_norm = (time.perf_counter() - t0) * 1000.0

        # Step 4: PPO inference
        t0 = time.perf_counter()
        try:
            import torch

            with torch.inference_mode():
                action, _raw_action = self._predict(norm_state)
        except ImportError:
            action, _raw_action = self._predict(norm_state)
        decoded = self._decode_action(action, workload, workload_dict)
        t_infer = (time.perf_counter() - t0) * 1000.0

        # Step 5: cost/carbon calculations
        t0 = time.perf_counter()
        decoded_dict = self._normalise_decoded_payload(decoded)
        cost_per_hr = self._estimate_cost_cached(decoded_dict, pricing)
        cost_savings = self._cost_savings_cached(decoded_dict, pricing)
        carbon_savings = self._carbon_savings_cached(decoded_dict, carbon)
        t_calc = (time.perf_counter() - t0) * 1000.0

        latency_ms = (time.perf_counter() - t_total) * 1000.0

        # Step 6: SHAP only when requested
        explanation: Dict[str, Any] = {}
        if include_explanation and self._explainer is not None and self._formatter is not None:
            try:
                built = self._build_explanation(
                    raw_state=state,
                    norm_state=norm_state,
                    action=action,
                    decoded=decoded,
                    workload=workload,
                    workload_dict=workload_dict,
                )
                if built is not None:
                    explanation = built if isinstance(built, dict) else {"detail": built}
            except Exception as exc:
                logger.warning("SchedulerAgent: SHAP explain failed: %s", exc)

        logger.debug(
            "SchedulerAgent.decide: total=%.1fms [data=%.1f state=%.1f norm=%.1f infer=%.1f calc=%.1f]",
            latency_ms, t_data, t_state, t_norm, t_infer, t_calc,
        )
        if t_infer > self._slow_infer_warn_ms:
            logger.warning("SchedulerAgent: slow inference %.0fms", t_infer)
        if t_data > self._slow_data_warn_ms:
            logger.warning("SchedulerAgent: slow data load %.0fms", t_data)

        result = self._merge_decision_output(
            decoded=decoded,
            workload_dict=workload_dict,
            action=action,
            explanation=explanation,
            duration_ms=latency_ms,
        )

        result["estimated_cost_per_hr"] = round(cost_per_hr, 4)
        result["cost_savings_pct"] = round(cost_savings, 2)
        result["carbon_savings_pct"] = round(carbon_savings, 2)
        result["latency_ms"] = round(latency_ms, 2)
        result["explanation"] = explanation
        result["_state"] = state
        result["_decoded"] = dict(decoded_dict)

        return result

    def predict_decision(
        self,
        workload: Any,
        include_explanation: bool = True,
    ) -> Dict[str, Any]:
        return self.schedule(workload=workload, include_explanation=include_explanation)

    def compute_explanation(self, state: np.ndarray, decoded: Dict) -> Dict:
        """
        Compute SHAP explanation in a guarded background-safe way.
        """
        if self._explainer is None:
            logger.warning("compute_explanation: _explainer is None — SHAP not loaded")
            return {}

        if self._formatter is None:
            logger.warning("compute_explanation: _formatter is None — formatter not loaded")
            return {}

        try:
            state = np.asarray(state, dtype=np.float32)
            if state.ndim > 1:
                state = state.reshape(-1)
        except Exception as exc:
            logger.error("compute_explanation: invalid state input: %s", exc, exc_info=True)
            return {}

        decoded_dict = self._normalise_decoded_payload(decoded)

        result_holder: Dict[str, Dict] = {}
        exception_holder: List[Exception] = []

        def _run() -> None:
            try:
                raw = self._explainer.explain(state)
                if not raw or raw.get("error"):
                    logger.warning(
                        "compute_explanation: explain() returned error or empty: %s",
                        raw.get("error", "empty dict") if raw else "None",
                    )
                    result_holder["result"] = {}
                    return

                formatted = self._formatter.format(raw, decoded_dict)
                result_holder["result"] = formatted

                logger.debug(
                    "compute_explanation: success — confidence=%.3f drivers=%d ms=%.0f",
                    float(formatted.get("confidence", 0.0) or 0.0),
                    len(formatted.get("top_drivers", [])),
                    float(formatted.get("explanation_ms", 0.0) or 0.0),
                )
            except Exception as exc:
                exception_holder.append(exc)
                logger.error(
                    "compute_explanation: exception during SHAP: %s",
                    exc,
                    exc_info=True,
                )

        worker = threading.Thread(target=_run, daemon=True)
        worker.start()
        worker.join(timeout=120.0)

        if worker.is_alive():
            logger.error(
                "compute_explanation: SHAP timed out after 120s for state shape %s",
                state.shape,
            )
            return {}

        if exception_holder:
            return {}

        return result_holder.get("result", {})

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    def _to_dict(self, obj: Any) -> Dict[str, Any]:
        if obj is None:
            return {}

        if isinstance(obj, dict):
            return dict(obj)

        if hasattr(obj, "model_dump"):
            try:
                return obj.model_dump()
            except Exception:
                pass

        if hasattr(obj, "dict"):
            try:
                return obj.dict()
            except Exception:
                pass

        if hasattr(obj, "__dict__"):
            try:
                return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
            except Exception:
                pass

        return {"value": obj}

    def _normalise_decoded_payload(self, decoded: Any) -> Dict[str, Any]:
        """
        Convert decoded action output into a plain dict for internal use.
        """
        if isinstance(decoded, dict):
            return dict(decoded)

        if hasattr(decoded, "model_dump"):
            try:
                data = decoded.model_dump()
                if isinstance(data, dict):
                    return dict(data)
            except Exception:
                pass

        if hasattr(decoded, "dict"):
            try:
                data = decoded.dict()
                if isinstance(data, dict):
                    return dict(data)
            except Exception:
                pass

        if hasattr(decoded, "__dict__"):
            try:
                return {k: v for k, v in vars(decoded).items() if not k.startswith("_")}
            except Exception:
                pass

        return {"decision": str(decoded)}

    def _load_carbon(self) -> Dict[str, float]:
        """
        Read carbon intensity data with a TTL cache and stampede protection.
        Only one thread refreshes stale data; others get warm or stale data.
        """
        global _carbon_cache_data, _carbon_cache_ts, _carbon_cache_refreshing

        now = time.monotonic()

        # Fast path without lock
        if _carbon_cache_data and (now - _carbon_cache_ts) < _CARBON_CACHE_TTL:
            return dict(_carbon_cache_data)

        with _carbon_cache_lock:
            now = time.monotonic()

            if _carbon_cache_data and (now - _carbon_cache_ts) < _CARBON_CACHE_TTL:
                return dict(_carbon_cache_data)

            if _carbon_cache_refreshing:
                return dict(_carbon_cache_data) if _carbon_cache_data else dict(self._STATIC_CARBON)

            _carbon_cache_refreshing = True

        path = (
            self._config.get("data_pipeline", {}).get("carbon_output_path")
            or self._config.get("environment", {}).get("carbon_output_path")
            or "data/carbon/carbon_intensity.json"
        )

        try:
            carbon_path = Path(path)
            result: Dict[str, float] = {}

            if carbon_path.exists():
                with open(carbon_path, "r", encoding="utf-8") as fh:
                    raw = json.load(fh) or {}

                for region, entry in raw.items():
                    if isinstance(entry, dict):
                        result[region] = float(entry.get("gco2_per_kwh", 415.0))
                    else:
                        result[region] = float(entry)

            if not result:
                result = dict(self._STATIC_CARBON)

            with _carbon_cache_lock:
                _carbon_cache_data = dict(result)
                _carbon_cache_ts = time.monotonic()
                _carbon_cache_refreshing = False

            return dict(result)

        except Exception as exc:
            logger.warning("SchedulerAgent._load_carbon: %s — using static", exc)

            with _carbon_cache_lock:
                if not _carbon_cache_data:
                    _carbon_cache_data = dict(self._STATIC_CARBON)
                    _carbon_cache_ts = time.monotonic()
                _carbon_cache_refreshing = False
                return dict(_carbon_cache_data)

    def build_state(self, workload: Dict[str, Any]) -> np.ndarray:
        """
        Public entry point — builds 45-dim state from workload dict.
        """
        workload_dict = self._to_dict(workload)
        pricing = self._get_pricing()
        carbon = self._load_carbon()
        return self._build_state_with(workload_dict, pricing, carbon)

    def _build_state_with(
        self,
        workload: Dict[str, Any],
        pricing: Dict[str, Any],
        carbon: Dict[str, Any],
    ) -> np.ndarray:
        """
        Calls StateBuilder.build(workload, pricing, carbon, history).
        Falls back carefully across older method signatures.
        """
        workload_dict = self._to_dict(workload)

        if "cpu_request_vcpu" in workload_dict and "cpu_request" not in workload_dict:
            workload_dict["cpu_request"] = workload_dict["cpu_request_vcpu"]

        history: List[Any] = []

        if self._state_builder is not None:
            try:
                state = self._state_builder.build(workload_dict, pricing, carbon, history)
                if isinstance(state, np.ndarray):
                    return state.astype(np.float32)
                return np.asarray(state, dtype=np.float32)
            except TypeError:
                pass
            except Exception as exc:
                logger.error("SchedulerAgent._build_state_with: build() failed: %s", exc, exc_info=True)

            try:
                state = self._state_builder.build(workload_dict)
                if isinstance(state, np.ndarray):
                    return state.astype(np.float32)
                return np.asarray(state, dtype=np.float32)
            except TypeError:
                pass
            except Exception as exc:
                logger.warning("_build_state_with: fallback build(workload) failed (%s)", exc)

            for method_name in ("build_state", "from_workload", "get_state"):
                method = getattr(self._state_builder, method_name, None)
                if callable(method):
                    try:
                        state = method(workload_dict)
                        if isinstance(state, np.ndarray):
                            return state.astype(np.float32)
                        return np.asarray(state, dtype=np.float32)
                    except Exception:
                        continue

        logger.error(
            "SchedulerAgent: no compatible StateBuilder method found. Available methods: %s",
            [m for m in dir(self._state_builder) if not m.startswith("_")]
            if self._state_builder
            else "state_builder=None",
        )
        return np.zeros(45, dtype=np.float32)

    def _build_state(self, workload: Dict[str, Any]) -> np.ndarray:
        workload_dict = self._to_dict(workload)
        pricing = self._get_pricing()
        carbon = self._load_carbon()
        return self._build_state_with(workload_dict, pricing, carbon)

    def _ensure_state_array(self, state: Any) -> np.ndarray:
        arr = np.asarray(state, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.reshape(-1)
        return arr

    def _normalise_obs(self, state: np.ndarray) -> np.ndarray:
        """
        Fast path:
          use pre-extracted VecNormalize stats and apply the same transform
          without calling vec_env.normalize_obs() on every request.

        Safe fallback:
          if stats are unavailable, use vec_env.normalize_obs().
        """
        state = np.asarray(state, dtype=np.float32)

        if self._has_vecnorm_stats and self._obs_mean is not None and self._obs_var is not None:
            try:
                normed = (state - self._obs_mean) / np.sqrt(self._obs_var + 1e-8)
                normed = np.clip(normed, -self._obs_clip, self._obs_clip)
                return normed.astype(np.float32)
            except Exception as exc:
                logger.warning(
                    "SchedulerAgent: lock-free normalization failed (%s) — falling back to vec_env.normalize_obs().",
                    exc,
                )

        if self._vec_env is None:
            return state

        try:
            batched = np.asarray([state], dtype=np.float32)
            normed = self._vec_env.normalize_obs(batched)

            if isinstance(normed, np.ndarray):
                if normed.ndim >= 2:
                    return normed[0].astype(np.float32)
                return normed.astype(np.float32)

            return state
        except Exception as exc:
            logger.warning(
                "SchedulerAgent: observation normalization failed (%s) — using raw state.",
                exc,
                exc_info=True,
            )
            return state

    def _predict(self, obs: np.ndarray) -> Tuple[List[int], Any]:
        if self._model is None:
            raise RuntimeError("SchedulerAgent: model not loaded")

        try:
            action, state = self._model.predict(obs, deterministic=True)
        except Exception as exc:
            logger.error("SchedulerAgent: PPO.predict failed: %s", exc, exc_info=True)
            raise

        action_list = self._as_action_list(action)
        return action_list, state

    def _as_action_list(self, action: Any) -> List[int]:
        if isinstance(action, np.ndarray):
            flat = action.reshape(-1).tolist()
            return [int(x) for x in flat]

        if isinstance(action, (list, tuple)):
            return [int(x) for x in action]

        return [int(action)]

    def _decode_action(
        self,
        action: List[int],
        workload: Any,
        workload_dict: Dict[str, Any],
    ) -> Any:
        candidates = [
            ("decode", (action,)),
            ("decode", (action, workload)),
            ("decode", (action, workload_dict)),
            ("decode_action", (action,)),
            ("decode_action", (action, workload)),
            ("decode_action", (action, workload_dict)),
        ]

        for method_name, args in candidates:
            method = getattr(self._decoder, method_name, None)
            if callable(method):
                try:
                    decoded = method(*args)
                    logger.debug(
                        "SchedulerAgent._decode_action: used ActionDecoder.%s",
                        method_name,
                    )
                    return decoded
                except TypeError:
                    continue

        logger.warning(
            "SchedulerAgent: no compatible ActionDecoder method found; returning raw action wrapper."
        )
        return {"action": action}

    def _build_explanation(
        self,
        raw_state: np.ndarray,
        norm_state: np.ndarray,
        action: List[int],
        decoded: Any,
        workload: Any,
        workload_dict: Dict[str, Any],
    ) -> Optional[Any]:
        if self._explainer is None:
            return None

        shap_result = None

        explain_candidates = [
            ("explain", (), {"state": norm_state}),
            ("explain", (), {"state": raw_state}),
            ("explain", (), {"observation": norm_state}),
            ("explain", (), {"observation": raw_state}),
            ("explain", (), {"obs": norm_state}),
            ("explain", (), {"obs": raw_state}),
            ("explain", (norm_state,), {}),
            ("explain", (raw_state,), {}),
        ]

        for method_name, args, kwargs in explain_candidates:
            method = getattr(self._explainer, method_name, None)
            if callable(method):
                try:
                    shap_result = method(*args, **kwargs)
                    break
                except TypeError:
                    continue
                except Exception as exc:
                    logger.warning(
                        "SchedulerAgent: explainer.%s failed (%s)",
                        method_name,
                        exc,
                        exc_info=True,
                    )
                    shap_result = None
                    break

        if shap_result is None:
            return None

        if self._formatter is None:
            return shap_result

        format_candidates = [
            ("format", (), {"explanation": shap_result, "decision": decoded, "workload": workload_dict}),
            ("format", (), {"explanation": shap_result, "decision": decoded}),
            ("format", (shap_result,), {}),
        ]

        for method_name, args, kwargs in format_candidates:
            method = getattr(self._formatter, method_name, None)
            if callable(method):
                try:
                    return method(*args, **kwargs)
                except TypeError:
                    continue
                except Exception as exc:
                    logger.warning(
                        "SchedulerAgent: formatter.%s failed (%s)",
                        method_name,
                        exc,
                        exc_info=True,
                    )
                    return shap_result

        return shap_result

    def _merge_decision_output(
        self,
        decoded: Any,
        workload_dict: Dict[str, Any],
        action: List[int],
        explanation: Any,
        duration_ms: float,
    ) -> Dict[str, Any]:
        """
        Normalize the final result into a dict that API / Kafka can serialize.
        """
        base: Dict[str, Any]

        if isinstance(decoded, dict):
            base = dict(decoded)
        elif hasattr(decoded, "model_dump"):
            try:
                base = decoded.model_dump()
            except Exception:
                base = {"decision": str(decoded)}
        elif hasattr(decoded, "dict"):
            try:
                base = decoded.dict()
            except Exception:
                base = {"decision": str(decoded)}
        elif hasattr(decoded, "__dict__"):
            try:
                base = {k: v for k, v in vars(decoded).items() if not k.startswith("_")}
            except Exception:
                base = {"decision": str(decoded)}
        else:
            base = {"decision": decoded}

        base.setdefault("workload", workload_dict)
        base.setdefault("action", action)
        base["inference_ms"] = duration_ms

        estimated = base.get("estimated_cost_per_hr")
        if estimated in (None, "", 0, 0.0):
            base["estimated_cost_per_hr"] = self.estimate_cost_per_hr(base)

        cost_savings = base.get("cost_savings_pct")
        if cost_savings in (None, "", 0, 0.0):
            base["cost_savings_pct"] = self.cost_savings_pct(base)

        carbon_savings = base.get("carbon_savings_pct")
        if carbon_savings in (None, "", 0, 0.0):
            base["carbon_savings_pct"] = self.carbon_savings_pct(base)

        if explanation is not None:
            base["explanation"] = explanation

        return base

    # -------------------------------------------------------------------------
    # Convenience API
    # -------------------------------------------------------------------------
    def warmup(self) -> None:
        try:
            dummy = np.zeros(45, dtype=np.float32)
            _ = self._normalise_obs(dummy)
            logger.info("SchedulerAgent: warmup complete")
        except Exception as exc:
            logger.warning("SchedulerAgent: warmup skipped (%s)", exc)

    def status(self) -> Dict[str, Any]:
        return {
            "agent_loaded": self._model is not None,
            "vecnorm_loaded": self._vec_env is not None,
            "vecnorm_stats_loaded": self._has_vecnorm_stats,
            "shap_ready": self._explainer is not None,
            "model_type": type(self._model).__name__ if self._model is not None else None,
        }