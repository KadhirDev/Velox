"""
CloudOS-RL Gymnasium Environment
=================================
Observation space: Box(45,) float32
Action space:      MultiDiscrete([4, 10, 10, 4, 6, 6]) → 57,600 combinations
                   [cloud, region, instance, scaling, purchase, sla_tier]

Carbon data loading priority:
  1. data/carbon/carbon_intensity.json  <- written by DataPipelineOrchestrator
     Reloaded per episode via file-mtime check (cheap stat() call, no re-parse
     unless the pipeline has actually updated the file)
  2. _STATIC_CARBON hardcoded values    <- always available, no file dependency

This ensures training works immediately without needing to run the pipeline first.
If the pipeline is running alongside training, the environment will automatically
pick up fresher carbon data at the start of each new episode.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ai_engine.environment.state_builder import StateBuilder
from ai_engine.environment.reward import RewardFunction
from ai_engine.environment.action_decoder import ActionDecoder
from ai_engine.cloud_adapter.pricing_cache import PricingCache

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Static fallback carbon (gCO2/kWh)
# Matches the 10-region subset used in StateBuilder state[20:30]
# ---------------------------------------------------------------------------

_STATIC_CARBON: Dict[str, float] = {
    "us-east-1":      415.0,
    "us-west-2":      192.0,
    "eu-west-1":      316.0,
    "eu-central-1":   338.0,
    "ap-southeast-1": 453.0,
    "ap-northeast-1": 506.0,
    "us-central1":    360.0,   # GCP alias
    "europe-west4":   284.0,   # GCP alias
    "eastus":         400.0,   # Azure alias
    "westeurope":     290.0,   # Azure alias
}

_CARBON_FILE = Path("data/carbon/carbon_intensity.json")


def _load_carbon_from_file() -> Dict[str, float]:
    """
    Reads pipeline-written carbon data if available.
    Returns a merged dict (pipeline data overrides static; static fills gaps).
    """
    try:
        if not _CARBON_FILE.exists():
            return dict(_STATIC_CARBON)

        with open(_CARBON_FILE) as fh:
            raw = json.load(fh)

        pipeline_carbon: Dict[str, float] = {}
        for region, entry in raw.items():
            val = entry.get("gco2_per_kwh") or entry.get("carbon_intensity_gco2_per_kwh")
            if val is not None:
                pipeline_carbon[region] = float(val)

        # Merge: static provides aliases that the pipeline may not cover
        return {**_STATIC_CARBON, **pipeline_carbon}

    except (json.JSONDecodeError, OSError, KeyError, TypeError) as exc:
        logger.warning("VeloxEnv: carbon file read failed (%s) — using static values.", exc)
        return dict(_STATIC_CARBON)


class VeloxEnv(gym.Env):
    """
    CloudOS-RL Gymnasium training and evaluation environment.
    """

    metadata    = {"render_modes": ["human"]}
    STATE_DIM   = 45
    ACTION_DIMS = [4, 10, 10, 4, 6, 6]

    def __init__(self, config: Dict, render_mode: Optional[str] = None):
        super().__init__()
        self.config      = config
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.STATE_DIM,), dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete(self.ACTION_DIMS)

        self._state_builder  = StateBuilder(config)
        self._reward_fn      = RewardFunction(config)
        self._action_decoder = ActionDecoder()
        self._pricing_cache  = PricingCache(config)

        # Load carbon at init; cheaply refreshed per episode via mtime check
        self._carbon:          Dict[str, float] = _load_carbon_from_file()
        self._carbon_mtime:    float = (
            _CARBON_FILE.stat().st_mtime if _CARBON_FILE.exists() else 0.0
        )

        self._step:     int              = 0
        self._max_steps: int             = config.get("max_episode_steps", 1000)
        self._state:    Optional[np.ndarray] = None
        self._history:  List[Dict]       = []

    # -----------------------------------------------------------------------
    # Gymnasium API
    # -----------------------------------------------------------------------

    def reset(
        self,
        seed:    Optional[int]  = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self._step = 0
        self._history.clear()
        self._maybe_refresh_carbon()
        self._state = self._build_state()
        return self._state, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self._step += 1

        decoded            = self._action_decoder.decode(action)
        reward, components = self._reward_fn.compute(
            action=decoded,
            state=self._state,
            pricing=self._pricing_cache.get_current_pricing(),
        )

        self._history.append({"reward": reward, "components": components})
        self._state  = self._build_state()
        terminated   = self._step >= self._max_steps

        info = {
            "step":              self._step,
            "decoded_action":    decoded,
            "reward_components": components,
        }
        return self._state, reward, terminated, False, info

    def render(self):
        if self.render_mode == "human" and self._history:
            c = self._history[-1]["components"]
            print(
                f"Step {self._step:4d} | "
                f"cost={c['cost']:+.3f}  "
                f"latency={c['latency']:+.3f}  "
                f"carbon={c['carbon']:+.3f}  "
                f"total={c['total']:+.3f}"
            )

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------

    def _build_state(self) -> np.ndarray:
        return self._state_builder.build(
            workload=self._sample_workload(),
            pricing=self._pricing_cache.get_current_pricing(),
            carbon=self._noisy_carbon(),
            history=self._history,
        )

    def _sample_workload(self) -> Dict:
        rng = self.np_random
        return {
            "cpu_request":             float(rng.uniform(0.25, 32.0)),
            "memory_request_gb":       float(rng.uniform(0.5, 128.0)),
            "gpu_count":               int(rng.choice([0, 0, 0, 1, 2, 4])),
            "storage_gb":              float(rng.uniform(10.0, 1000.0)),
            "network_bandwidth_gbps":  float(rng.uniform(0.1, 10.0)),
            "expected_duration_hours": float(rng.uniform(0.1, 720.0)),
            "priority":                int(rng.integers(1, 5)),
            "sla_latency_ms":          float(rng.choice([10, 50, 100, 200, 500])),
            "workload_type":           str(rng.choice(["batch", "realtime", "ml_training", "web_service"])),
            "is_spot_tolerant":        bool(rng.choice([True, False])),
        }

    def _noisy_carbon(self) -> Dict[str, float]:
        """
        Adds small Gaussian noise to carbon values to simulate
        real-world grid mix variability within an episode.
        """
        noise = self.np_random.normal(0, 8.0, len(self._carbon))
        return {
            k: max(10.0, v + float(n))
            for (k, v), n in zip(self._carbon.items(), noise)
        }

    def _maybe_refresh_carbon(self):
        """
        Reloads carbon from file only if the pipeline has written a newer version.
        Uses a stat() mtime check — does NOT re-parse JSON unless file changed.
        Called at the start of each episode (reset).
        """
        if not _CARBON_FILE.exists():
            return
        try:
            mtime = _CARBON_FILE.stat().st_mtime
            if mtime > self._carbon_mtime:
                self._carbon       = _load_carbon_from_file()
                self._carbon_mtime = mtime
                logger.debug("VeloxEnv: reloaded carbon from updated pipeline file.")
        except OSError:
            pass
