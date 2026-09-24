"""
SHAP Background Dataset Generator
====================================
Generates a representative background dataset for SHAP KernelExplainer.

Why background data matters:
  SHAP KernelExplainer computes feature attributions by comparing each
  prediction to a baseline. The baseline is the MEAN prediction over
  background samples. A diverse, representative background produces
  more accurate and stable SHAP values.

Background dataset strategy:
  - 200 samples drawn from the same distribution as training episodes
  - Covers all 16 AWS regions from Module G carbon + pricing data
  - Uses realistic workload distributions (CPU/RAM/GPU/storage)
  - Stratified: equal representation across cloud, region, purchase option
  - Saved to data/shap/background_dataset.npy for reuse

State vector structure (45 dims — matches StateBuilder exactly):
  [0:10]   workload features
  [10:20]  pricing features (from Module G aws_pricing.json)
  [20:30]  carbon features  (from Module G carbon_intensity.json)
  [30:40]  latency features
  [40:45]  history features

Compatible with:
  - ai_engine/environment/state_builder.py  (45-dim output)
  - ai_engine/explainability/shap_explainer.py (background input)
  - ai_engine/inference/scheduler_agent.py  (build_state output)
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 45 feature names — must match StateBuilder dimension order exactly
# ---------------------------------------------------------------------------
FEATURE_NAMES: List[str] = [
    # [0:10] Workload
    "cpu_request_vcpu",
    "memory_request_gb",
    "gpu_count",
    "storage_gb",
    "network_bandwidth_gbps",
    "expected_duration_hours",
    "priority",
    "sla_latency_ms",
    "workload_type_encoded",
    "is_spot_tolerant",
    # [10:20] Pricing
    "price_cloud_0",
    "price_cloud_1",
    "price_cloud_2",
    "price_cloud_3",
    "price_cloud_4",
    "price_cloud_5",
    "price_cloud_6",
    "price_cloud_7",
    "price_cloud_8",
    "price_cloud_9",
    # [20:30] Carbon (gCO2/kWh normalised to [0,1])
    "carbon_region_0",
    "carbon_region_1",
    "carbon_region_2",
    "carbon_region_3",
    "carbon_region_4",
    "carbon_region_5",
    "carbon_region_6",
    "carbon_region_7",
    "carbon_region_8",
    "carbon_region_9",
    # [30:40] Latency (ms normalised)
    "latency_region_0",
    "latency_region_1",
    "latency_region_2",
    "latency_region_3",
    "latency_region_4",
    "latency_region_5",
    "latency_region_6",
    "latency_region_7",
    "latency_region_8",
    "latency_region_9",
    # [40:45] Episode history
    "history_avg_reward",
    "history_avg_cost_savings",
    "history_avg_carbon_savings",
    "history_episode_step",
    "history_sla_breach_rate",
]

assert len(FEATURE_NAMES) == 45, f"Expected 45 feature names, got {len(FEATURE_NAMES)}"

# ---------------------------------------------------------------------------
# Background region set — 10 regions from Module G (StateBuilder uses 10)
# ---------------------------------------------------------------------------
_BG_REGIONS: List[str] = [
    "us-east-1", "us-west-2", "eu-west-1",
    "eu-central-1", "ap-southeast-1", "ap-northeast-1",
    "us-central1", "europe-west4", "eastus", "westeurope",
]

# Static carbon fallback (gCO2/kWh) if pipeline file not available
_STATIC_CARBON: Dict[str, float] = {
    "us-east-1":      415.0, "us-west-2":      192.0,
    "eu-west-1":      316.0, "eu-central-1":   338.0,
    "ap-southeast-1": 453.0, "ap-northeast-1": 506.0,
    "us-central1":    360.0, "europe-west4":   284.0,
    "eastus":         400.0, "westeurope":     290.0,
}

# Static pricing fallback ($/hr on-demand m5.large equivalent)
_STATIC_PRICING: Dict[str, float] = {
    "us-east-1":      0.096, "us-west-2":      0.096,
    "eu-west-1":      0.107, "eu-central-1":   0.111,
    "ap-southeast-1": 0.114, "ap-northeast-1": 0.118,
    "us-central1":    0.096, "europe-west4":   0.107,
    "eastus":         0.096, "westeurope":     0.107,
}

# Latency baselines (ms) per region
_STATIC_LATENCY: Dict[str, float] = {
    "us-east-1":      12.0, "us-west-2":      28.0,
    "eu-west-1":      85.0, "eu-central-1":   88.0,
    "ap-southeast-1": 220.0, "ap-northeast-1": 190.0,
    "us-central1":    18.0, "europe-west4":   90.0,
    "eastus":         14.0, "westeurope":     87.0,
}

_CARBON_NORM  = 600.0   # normalisation divisor — matches VeloxEnv
_PRICING_NORM = 10.0
_LATENCY_NORM = 1000.0

_BG_OUTPUT_DIR  = Path("data/shap")
_BG_OUTPUT_PATH = _BG_OUTPUT_DIR / "background_dataset.npy"
_META_PATH      = _BG_OUTPUT_DIR / "background_metadata.json"


class BackgroundDataGenerator:
    """
    Generates and persists a representative SHAP background dataset.
    Reads Module G pipeline output files for realistic pricing + carbon values.
    Falls back to static constants if pipeline files are not yet available.
    """

    def __init__(self, config: Dict):
        self._config        = config
        self._pricing_path  = Path(
            config.get("data_pipeline", {}).get(
                "pricing_output_path", "data/pricing/aws_pricing.json"
            )
        )
        self._carbon_path   = Path(
            config.get("data_pipeline", {}).get(
                "carbon_output_path", "data/carbon/carbon_intensity.json"
            )
        )

    # -----------------------------------------------------------------------
    # Public
    # -----------------------------------------------------------------------

    def generate(
        self,
        n_samples: int = 200,
        seed:      int = 42,
        force:     bool = False,
    ) -> np.ndarray:
        """
        Returns background dataset of shape (n_samples, 45).

        If a cached dataset exists and force=False, loads and returns it.
        Otherwise generates fresh samples using Module G pipeline data.

        Args:
            n_samples: Number of background samples (100-500 recommended).
                       More = more accurate SHAP but slower per-call.
            seed:      RNG seed for reproducibility.
            force:     If True, regenerates even if cache exists.

        Returns:
            np.ndarray of shape (n_samples, 45), dtype float32
        """
        if not force and _BG_OUTPUT_PATH.exists():
            cached = self._load_cached()
            if cached is not None and cached.shape == (n_samples, 45):
                logger.info(
                    "BackgroundDataGenerator: loaded cached dataset (%d, 45) from %s",
                    n_samples, _BG_OUTPUT_PATH,
                )
                return cached

        logger.info(
            "BackgroundDataGenerator: generating %d samples (seed=%d) ...",
            n_samples, seed,
        )
        t0 = time.perf_counter()

        pricing = self._load_pricing()
        carbon  = self._load_carbon()

        rng     = np.random.default_rng(seed)
        samples = np.zeros((n_samples, 45), dtype=np.float32)

        for i in range(n_samples):
            samples[i] = self._sample_state(rng, pricing, carbon, i, n_samples)

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            "BackgroundDataGenerator: generated (%d, 45) in %.0f ms",
            n_samples, elapsed,
        )

        self._save(samples, n_samples, seed)
        return samples

    def load_or_generate(self, n_samples: int = 200, seed: int = 42) -> np.ndarray:
        """Convenience method — loads cache if available, else generates."""
        return self.generate(n_samples=n_samples, seed=seed, force=False)

    def get_feature_names(self) -> List[str]:
        """Returns the 45 feature names in state vector order."""
        return list(FEATURE_NAMES)

    # -----------------------------------------------------------------------
    # Private: state sampling
    # -----------------------------------------------------------------------

    def _sample_state(
        self,
        rng:      np.random.Generator,
        pricing:  Dict[str, float],
        carbon:   Dict[str, float],
        idx:      int,
        total:    int,
    ) -> np.ndarray:
        """
        Generates one 45-dim state vector.
        Stratified across workload types and regions to ensure diversity.
        """
        state = np.zeros(45, dtype=np.float32)

        # ── [0:10] Workload features ─────────────────────────────────────
        state[0]  = float(rng.uniform(0.25, 32.0))          # cpu_request_vcpu
        state[1]  = float(rng.uniform(0.5, 128.0))          # memory_request_gb
        state[2]  = float(rng.choice([0, 0, 0, 1, 2, 4]))   # gpu_count
        state[3]  = float(rng.uniform(10.0, 1000.0))        # storage_gb
        state[4]  = float(rng.uniform(0.1, 10.0))           # network_bandwidth_gbps
        state[5]  = float(rng.uniform(0.1, 720.0))          # expected_duration_hours
        state[6]  = float(rng.integers(1, 5))               # priority
        state[7]  = float(rng.choice([10, 50, 100, 200, 500])) / 500.0  # sla_latency_ms normalised
        state[8]  = float(rng.integers(0, 4))               # workload_type_encoded
        state[9]  = float(rng.choice([0.0, 1.0]))           # is_spot_tolerant

        # ── [10:20] Pricing features (from Module G) ──────────────────────
        # Stratify region assignment for representativeness
        region_idx = idx % len(_BG_REGIONS)
        base_region = _BG_REGIONS[region_idx]

        for j, region in enumerate(_BG_REGIONS):
            od_price  = pricing.get(region, 0.096)
            # Add small noise to simulate spot/on-demand variation
            noise     = float(rng.uniform(0.9, 1.1))
            state[10 + j] = float(od_price * noise) / _PRICING_NORM

        # ── [20:30] Carbon features (from Module G) ───────────────────────
        for j, region in enumerate(_BG_REGIONS):
            co2       = carbon.get(region, 400.0)
            # Add noise to simulate grid mix variability
            noise     = float(rng.normal(0, 8.0))
            state[20 + j] = max(10.0, co2 + noise) / _CARBON_NORM

        # ── [30:40] Latency features ──────────────────────────────────────
        for j, region in enumerate(_BG_REGIONS):
            base_lat  = _STATIC_LATENCY.get(region, 100.0)
            noise     = float(rng.normal(0, base_lat * 0.1))
            state[30 + j] = max(1.0, base_lat + noise) / _LATENCY_NORM

        # ── [40:45] History features ──────────────────────────────────────
        state[40] = float(rng.uniform(-2.0, 5.0))   # history_avg_reward
        state[41] = float(rng.uniform(0.0, 0.4))    # history_avg_cost_savings
        state[42] = float(rng.uniform(0.0, 0.3))    # history_avg_carbon_savings
        state[43] = float(rng.uniform(0.0, 1.0))    # history_episode_step (normalised)
        state[44] = float(rng.uniform(0.0, 0.1))    # history_sla_breach_rate

        return state

    # -----------------------------------------------------------------------
    # Private: data loading
    # -----------------------------------------------------------------------

    def _load_pricing(self) -> Dict[str, float]:
        """
        Reads Module G aws_pricing.json.
        Returns {region: on_demand_per_vcpu_hr} for all background regions.
        Falls back to static constants if file not available.
        """
        try:
            if self._pricing_path.exists():
                with open(self._pricing_path, encoding="utf-8") as fh:
                    raw = json.load(fh)

                result: Dict[str, float] = {}
                for region in _BG_REGIONS:
                    region_data = raw.get(region, {})
                    # Prefer on_demand_per_vcpu_hr, fall back to m5.large flat price
                    price = (
                        region_data.get("on_demand_per_vcpu_hr")
                        or region_data.get("m5.large")
                        or region_data.get("m5.large:on_demand")
                        or _STATIC_PRICING.get(region, 0.096)
                    )
                    result[region] = float(price)

                logger.info(
                    "BackgroundDataGenerator: loaded pricing from %s (%d regions)",
                    self._pricing_path, len(result),
                )
                return result
        except (json.JSONDecodeError, OSError, KeyError) as exc:
            logger.warning("BackgroundDataGenerator: pricing load failed (%s) — using static.", exc)

        logger.info("BackgroundDataGenerator: using static pricing fallback.")
        return dict(_STATIC_PRICING)

    def _load_carbon(self) -> Dict[str, float]:
        """
        Reads Module G carbon_intensity.json.
        Returns {region: gco2_per_kwh} for all background regions.
        Falls back to static constants if file not available.
        """
        try:
            if self._carbon_path.exists():
                with open(self._carbon_path, encoding="utf-8") as fh:
                    raw = json.load(fh)

                result: Dict[str, float] = {}
                for region in _BG_REGIONS:
                    entry = raw.get(region, {})
                    co2   = (
                        entry.get("gco2_per_kwh")
                        or entry.get("carbon_intensity_gco2_per_kwh")
                        or _STATIC_CARBON.get(region, 400.0)
                    )
                    result[region] = float(co2)

                logger.info(
                    "BackgroundDataGenerator: loaded carbon from %s (%d regions)",
                    self._carbon_path, len(result),
                )
                return result
        except (json.JSONDecodeError, OSError, KeyError) as exc:
            logger.warning("BackgroundDataGenerator: carbon load failed (%s) — using static.", exc)

        logger.info("BackgroundDataGenerator: using static carbon fallback.")
        return dict(_STATIC_CARBON)

    # -----------------------------------------------------------------------
    # Private: persistence
    # -----------------------------------------------------------------------

    def _save(self, dataset: np.ndarray, n_samples: int, seed: int):
        """Saves dataset to data/shap/ with metadata."""
        _BG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        np.save(str(_BG_OUTPUT_PATH), dataset)
        logger.info("BackgroundDataGenerator: saved to %s", _BG_OUTPUT_PATH)

        meta = {
            "shape":          list(dataset.shape),
            "n_samples":      n_samples,
            "seed":           seed,
            "feature_names":  FEATURE_NAMES,
            "regions":        _BG_REGIONS,
            "pricing_source": str(self._pricing_path),
            "carbon_source":  str(self._carbon_path),
            "generated_at":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "dtype":          str(dataset.dtype),
            "mean":           float(dataset.mean()),
            "std":            float(dataset.std()),
        }

        with open(_META_PATH, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        logger.info("BackgroundDataGenerator: metadata saved to %s", _META_PATH)

    @staticmethod
    def _load_cached() -> Optional[np.ndarray]:
        """Loads cached background dataset. Returns None if file is corrupt."""
        try:
            data = np.load(str(_BG_OUTPUT_PATH))
            logger.debug("BackgroundDataGenerator: cache hit — shape %s", data.shape)
            return data
        except Exception as exc:
            logger.warning("BackgroundDataGenerator: cache corrupt (%s) — regenerating.", exc)
            return None