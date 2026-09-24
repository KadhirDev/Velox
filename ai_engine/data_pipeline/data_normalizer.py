"""
Data Normalizer
================
Merges raw outputs from all three fetchers and writes canonical JSON files.

Files written:
  data/pricing/aws_pricing.json        <- PricingCache reads on TTL expiry
  data/pricing/aws_actual_costs.json   <- reward calibration, anomaly reference
  data/carbon/carbon_intensity.json    <- VeloxEnv reads per-episode

DEFENSIVE BEHAVIOUR:
  If raw_pricing is empty (AWS API unavailable / IAM missing),
  the normalizer writes the hardcoded fallback pricing constants
  instead of writing {}. This ensures the file is always usable.

All writes are ATOMIC (write .tmp.json → rename) so readers
never see a partial file even if the writer crashes mid-write.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_PRICING_PATH = Path("data/pricing/aws_pricing.json")
_DEFAULT_ACTUAL_COSTS_PATH = Path("data/pricing/aws_actual_costs.json")
_DEFAULT_CARBON_PATH = Path("data/carbon/carbon_intensity.json")

# ---------------------------------------------------------------------------
# Hardcoded fallback pricing — written to file when AWS API is unavailable
# Keeps aws_pricing.json non-empty and usable by PricingCache at all times
# ---------------------------------------------------------------------------
_FALLBACK_PRICING: Dict[str, Dict] = {
    "us-east-1": {
        "t3.medium": 0.0416,
        "t3.medium:on_demand": 0.0416,
        "t3.medium:spot": 0.0137,
        "t3.large": 0.0832,
        "t3.large:on_demand": 0.0832,
        "t3.large:spot": 0.0275,
        "m5.large": 0.0960,
        "m5.large:on_demand": 0.0960,
        "m5.large:spot": 0.0317,
        "m5.xlarge": 0.1920,
        "m5.xlarge:on_demand": 0.1920,
        "m5.xlarge:spot": 0.0634,
        "c5.large": 0.0850,
        "c5.large:on_demand": 0.0850,
        "c5.large:spot": 0.0281,
        "c5.xlarge": 0.1700,
        "c5.xlarge:on_demand": 0.1700,
        "c5.xlarge:spot": 0.0561,
        "r5.large": 0.1260,
        "r5.large:on_demand": 0.1260,
        "r5.large:spot": 0.0416,
        "r5.xlarge": 0.2520,
        "r5.xlarge:on_demand": 0.2520,
        "r5.xlarge:spot": 0.0832,
        "g4dn.xlarge": 0.5260,
        "g4dn.xlarge:on_demand": 0.5260,
        "g4dn.xlarge:spot": 0.1736,
        "p3.2xlarge": 3.0600,
        "p3.2xlarge:on_demand": 3.0600,
        "p3.2xlarge:spot": 1.0098,
        "m5.large:savings_plan": 0.0528,
        "m5.large:reserved_1yr": 0.0576,
        "m5.large:reserved_3yr": 0.0384,
        "c5.large:savings_plan": 0.0468,
        "c5.large:reserved_1yr": 0.0510,
        "c5.large:reserved_3yr": 0.0340,
        "on_demand_per_vcpu_hr": 0.048,
        "_nested": {
            "m5.large": {
                "on_demand": 0.0960,
                "spot": 0.0317,
                "savings_plan": 0.0528,
                "reserved_1yr": 0.0576,
                "reserved_3yr": 0.0384,
            },
            "c5.large": {
                "on_demand": 0.0850,
                "spot": 0.0281,
                "savings_plan": 0.0468,
                "reserved_1yr": 0.0510,
                "reserved_3yr": 0.0340,
            },
            "r5.large": {
                "on_demand": 0.1260,
                "spot": 0.0416,
                "savings_plan": 0.0693,
                "reserved_1yr": 0.0756,
                "reserved_3yr": 0.0504,
            },
            "t3.medium": {
                "on_demand": 0.0416,
                "spot": 0.0137,
                "savings_plan": 0.0229,
                "reserved_1yr": 0.0250,
                "reserved_3yr": 0.0166,
            },
        },
        "_region": "us-east-1",
        "_updated": "static_fallback",
        "_source": "hardcoded_fallback",
    },
    "us-east-2": {
        "m5.large": 0.0960,
        "m5.large:on_demand": 0.0960,
        "m5.large:spot": 0.0317,
        "c5.large": 0.0850,
        "c5.large:on_demand": 0.0850,
        "c5.large:spot": 0.0281,
        "on_demand_per_vcpu_hr": 0.048,
        "_nested": {
            "m5.large": {
                "on_demand": 0.0960,
                "spot": 0.0317,
                "savings_plan": 0.0528,
                "reserved_1yr": 0.0576,
                "reserved_3yr": 0.0384,
            }
        },
        "_region": "us-east-2",
        "_source": "hardcoded_fallback",
    },
    "us-west-1": {
        "m5.large": 0.1120,
        "m5.large:on_demand": 0.1120,
        "m5.large:spot": 0.0370,
        "c5.large": 0.0960,
        "c5.large:on_demand": 0.0960,
        "c5.large:spot": 0.0317,
        "on_demand_per_vcpu_hr": 0.056,
        "_nested": {
            "m5.large": {
                "on_demand": 0.1120,
                "spot": 0.0370,
                "savings_plan": 0.0616,
                "reserved_1yr": 0.0672,
                "reserved_3yr": 0.0448,
            }
        },
        "_region": "us-west-1",
        "_source": "hardcoded_fallback",
    },
    "us-west-2": {
        "m5.large": 0.0960,
        "m5.large:on_demand": 0.0960,
        "m5.large:spot": 0.0317,
        "c5.large": 0.0850,
        "c5.large:on_demand": 0.0850,
        "c5.large:spot": 0.0281,
        "on_demand_per_vcpu_hr": 0.048,
        "_nested": {
            "m5.large": {
                "on_demand": 0.0960,
                "spot": 0.0317,
                "savings_plan": 0.0528,
                "reserved_1yr": 0.0576,
                "reserved_3yr": 0.0384,
            }
        },
        "_region": "us-west-2",
        "_source": "hardcoded_fallback",
    },
    "eu-west-1": {
        "m5.large": 0.1070,
        "m5.large:on_demand": 0.1070,
        "m5.large:spot": 0.0353,
        "c5.large": 0.0970,
        "c5.large:on_demand": 0.0970,
        "c5.large:spot": 0.0320,
        "on_demand_per_vcpu_hr": 0.054,
        "_nested": {
            "m5.large": {
                "on_demand": 0.1070,
                "spot": 0.0353,
                "savings_plan": 0.0589,
                "reserved_1yr": 0.0642,
                "reserved_3yr": 0.0428,
            }
        },
        "_region": "eu-west-1",
        "_source": "hardcoded_fallback",
    },
    "eu-central-1": {
        "m5.large": 0.1110,
        "m5.large:on_demand": 0.1110,
        "m5.large:spot": 0.0366,
        "c5.large": 0.0990,
        "c5.large:on_demand": 0.0990,
        "c5.large:spot": 0.0327,
        "on_demand_per_vcpu_hr": 0.056,
        "_nested": {
            "m5.large": {
                "on_demand": 0.1110,
                "spot": 0.0366,
                "savings_plan": 0.0611,
                "reserved_1yr": 0.0666,
                "reserved_3yr": 0.0444,
            }
        },
        "_region": "eu-central-1",
        "_source": "hardcoded_fallback",
    },
    "eu-north-1": {
        "m5.large": 0.0960,
        "m5.large:on_demand": 0.0960,
        "m5.large:spot": 0.0317,
        "c5.large": 0.0860,
        "c5.large:on_demand": 0.0860,
        "c5.large:spot": 0.0284,
        "on_demand_per_vcpu_hr": 0.048,
        "_nested": {
            "m5.large": {
                "on_demand": 0.0960,
                "spot": 0.0317,
                "savings_plan": 0.0528,
                "reserved_1yr": 0.0576,
                "reserved_3yr": 0.0384,
            }
        },
        "_region": "eu-north-1",
        "_source": "hardcoded_fallback",
    },
    "ap-southeast-1": {
        "m5.large": 0.1140,
        "m5.large:on_demand": 0.1140,
        "m5.large:spot": 0.0376,
        "c5.large": 0.1000,
        "c5.large:on_demand": 0.1000,
        "c5.large:spot": 0.0330,
        "on_demand_per_vcpu_hr": 0.057,
        "_nested": {
            "m5.large": {
                "on_demand": 0.1140,
                "spot": 0.0376,
                "savings_plan": 0.0627,
                "reserved_1yr": 0.0684,
                "reserved_3yr": 0.0456,
            }
        },
        "_region": "ap-southeast-1",
        "_source": "hardcoded_fallback",
    },
    "ap-northeast-1": {
        "m5.large": 0.1180,
        "m5.large:on_demand": 0.1180,
        "m5.large:spot": 0.0389,
        "c5.large": 0.1070,
        "c5.large:on_demand": 0.1070,
        "c5.large:spot": 0.0353,
        "on_demand_per_vcpu_hr": 0.059,
        "_nested": {
            "m5.large": {
                "on_demand": 0.1180,
                "spot": 0.0389,
                "savings_plan": 0.0649,
                "reserved_1yr": 0.0708,
                "reserved_3yr": 0.0472,
            }
        },
        "_region": "ap-northeast-1",
        "_source": "hardcoded_fallback",
    },
    "ca-central-1": {
        "m5.large": 0.1000,
        "m5.large:on_demand": 0.1000,
        "m5.large:spot": 0.0330,
        "c5.large": 0.0900,
        "c5.large:on_demand": 0.0900,
        "c5.large:spot": 0.0297,
        "on_demand_per_vcpu_hr": 0.050,
        "_nested": {
            "m5.large": {
                "on_demand": 0.1000,
                "spot": 0.0330,
                "savings_plan": 0.0550,
                "reserved_1yr": 0.0600,
                "reserved_3yr": 0.0400,
            }
        },
        "_region": "ca-central-1",
        "_source": "hardcoded_fallback",
    },
    "sa-east-1": {
        "m5.large": 0.1420,
        "m5.large:on_demand": 0.1420,
        "m5.large:spot": 0.0469,
        "c5.large": 0.1280,
        "c5.large:on_demand": 0.1280,
        "c5.large:spot": 0.0422,
        "on_demand_per_vcpu_hr": 0.071,
        "_nested": {
            "m5.large": {
                "on_demand": 0.1420,
                "spot": 0.0469,
                "savings_plan": 0.0781,
                "reserved_1yr": 0.0852,
                "reserved_3yr": 0.0568,
            }
        },
        "_region": "sa-east-1",
        "_source": "hardcoded_fallback",
    },
    "ap-south-1": {
        "m5.large": 0.0960,
        "m5.large:on_demand": 0.0960,
        "m5.large:spot": 0.0317,
        "c5.large": 0.0850,
        "c5.large:on_demand": 0.0850,
        "c5.large:spot": 0.0281,
        "on_demand_per_vcpu_hr": 0.048,
        "_nested": {
            "m5.large": {
                "on_demand": 0.0960,
                "spot": 0.0317,
                "savings_plan": 0.0528,
                "reserved_1yr": 0.0576,
                "reserved_3yr": 0.0384,
            }
        },
        "_region": "ap-south-1",
        "_source": "hardcoded_fallback",
    },
    "us-central1": {
        "m5.large": 0.0960,
        "m5.large:on_demand": 0.0960,
        "m5.large:spot": 0.0317,
        "on_demand_per_vcpu_hr": 0.048,
        "_nested": {
            "m5.large": {
                "on_demand": 0.0960,
                "spot": 0.0317,
                "savings_plan": 0.0528,
                "reserved_1yr": 0.0576,
                "reserved_3yr": 0.0384,
            }
        },
        "_region": "us-central1",
        "_source": "hardcoded_fallback",
    },
    "europe-west4": {
        "m5.large": 0.1070,
        "m5.large:on_demand": 0.1070,
        "m5.large:spot": 0.0353,
        "on_demand_per_vcpu_hr": 0.054,
        "_nested": {
            "m5.large": {
                "on_demand": 0.1070,
                "spot": 0.0353,
                "savings_plan": 0.0589,
                "reserved_1yr": 0.0642,
                "reserved_3yr": 0.0428,
            }
        },
        "_region": "europe-west4",
        "_source": "hardcoded_fallback",
    },
    "eastus": {
        "m5.large": 0.0960,
        "m5.large:on_demand": 0.0960,
        "m5.large:spot": 0.0317,
        "on_demand_per_vcpu_hr": 0.048,
        "_nested": {
            "m5.large": {
                "on_demand": 0.0960,
                "spot": 0.0317,
                "savings_plan": 0.0528,
                "reserved_1yr": 0.0576,
                "reserved_3yr": 0.0384,
            }
        },
        "_region": "eastus",
        "_source": "hardcoded_fallback",
    },
    "westeurope": {
        "m5.large": 0.1070,
        "m5.large:on_demand": 0.1070,
        "m5.large:spot": 0.0353,
        "on_demand_per_vcpu_hr": 0.054,
        "_nested": {
            "m5.large": {
                "on_demand": 0.1070,
                "spot": 0.0353,
                "savings_plan": 0.0589,
                "reserved_1yr": 0.0642,
                "reserved_3yr": 0.0428,
            }
        },
        "_region": "westeurope",
        "_source": "hardcoded_fallback",
    },
}

_INSTANCE_VCPU: Dict[str, int] = {
    "t3.medium": 2,
    "t3.large": 2,
    "m5.large": 2,
    "m5.xlarge": 4,
    "c5.large": 2,
    "c5.xlarge": 4,
    "r5.large": 2,
    "r5.xlarge": 4,
    "g4dn.xlarge": 4,
    "p3.2xlarge": 8,
}


class DataNormalizer:
    """
    Stateless transformer and atomic file writer.
    Safe to call from multiple threads.
    Always writes non-empty files — uses hardcoded fallback if AWS API unavailable.
    """

    def __init__(self, config: Dict):
        self._config = config
        dp = config.get("data_pipeline", {})
        self._pricing_path = Path(dp.get("pricing_output_path", str(_DEFAULT_PRICING_PATH)))
        self._actual_costs_path = Path(
            dp.get("actual_costs_output_path", str(_DEFAULT_ACTUAL_COSTS_PATH))
        )
        self._carbon_path = Path(dp.get("carbon_output_path", str(_DEFAULT_CARBON_PATH)))

    # -----------------------------------------------------------------------
    # Public: pricing
    # -----------------------------------------------------------------------

    def normalize_pricing(
        self,
        raw_pricing: Dict[str, Dict],
        cur_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict]:
        """
        Merges Pricing API data with optional CUR blended rates.
        DEFENSIVE: if raw_pricing is empty (IAM denied / no credentials),
        writes hardcoded fallback pricing so the file is always usable.
        Writes result to data/pricing/aws_pricing.json.
        Returns the merged dict.
        """
        if not raw_pricing:
            logger.warning(
                "DataNormalizer: raw_pricing is empty — writing hardcoded fallback pricing to %s. "
                "Fix IAM permission pricing:GetProducts to get real prices.",
                self._pricing_path,
            )
            self._atomic_write(self._pricing_path, _FALLBACK_PRICING)
            logger.info(
                "DataNormalizer: wrote pricing → %s (%d regions, source=fallback)",
                self._pricing_path,
                len(_FALLBACK_PRICING),
            )
            return _FALLBACK_PRICING

        blended_rates = (cur_data or {}).get("blended_rates", {})
        merged: Dict[str, Dict] = {}

        for region, instances in raw_pricing.items():
            region_entry: Dict[str, Any] = {}
            nested: Dict[str, Any] = {}
            vcpu_rates: List[float] = []

            for key, value in instances.items():
                if key.startswith("_") or key == "on_demand_per_vcpu_hr":
                    continue

                if ":" in key:
                    region_entry[key] = value
                    continue

                if not isinstance(value, (int, float)):
                    continue

                on_demand = float(value)

                blended = blended_rates.get(region, {}).get(key)
                calibrated = blended if blended else on_demand

                nested_prices = instances.get("_nested", {}).get(key, {})
                spot = nested_prices.get("spot", on_demand * 0.33)
                sav_plan = nested_prices.get("savings_plan", on_demand * 0.55)
                res_1yr = nested_prices.get("reserved_1yr", on_demand * 0.60)
                res_3yr = nested_prices.get("reserved_3yr", on_demand * 0.40)

                region_entry[key] = round(calibrated, 6)
                region_entry[f"{key}:on_demand"] = round(on_demand, 6)
                region_entry[f"{key}:spot"] = round(spot, 6)
                region_entry[f"{key}:savings_plan"] = round(sav_plan, 6)
                region_entry[f"{key}:reserved_1yr"] = round(res_1yr, 6)
                region_entry[f"{key}:reserved_3yr"] = round(res_3yr, 6)
                region_entry[f"{key}:blended"] = round(calibrated, 6)

                nested[key] = {
                    "on_demand": round(on_demand, 6),
                    "spot": round(spot, 6),
                    "savings_plan": round(sav_plan, 6),
                    "reserved_1yr": round(res_1yr, 6),
                    "reserved_3yr": round(res_3yr, 6),
                    "blended": round(calibrated, 6),
                }

                vcpus = _INSTANCE_VCPU.get(key, 2)
                vcpu_rates.append(on_demand / vcpus)

            if "on_demand_per_vcpu_hr" in instances:
                region_entry["on_demand_per_vcpu_hr"] = round(
                    float(instances["on_demand_per_vcpu_hr"]), 6
                )
            elif vcpu_rates:
                region_entry["on_demand_per_vcpu_hr"] = round(sum(vcpu_rates) / len(vcpu_rates), 6)

            region_entry["_nested"] = nested
            region_entry["_region"] = region
            region_entry["_updated"] = datetime.now(tz=timezone.utc).isoformat()
            merged[region] = region_entry

        self._atomic_write(self._pricing_path, merged)
        logger.info(
            "DataNormalizer: wrote pricing → %s (%d regions)",
            self._pricing_path,
            len(merged),
        )
        return merged

    # -----------------------------------------------------------------------
    # Public: actual costs
    # -----------------------------------------------------------------------

    def normalize_actual_costs(self, cur_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Writes CUR blended_rates + usage_summary to aws_actual_costs.json.
        Always writes a valid JSON file — even if CUR fetch failed.
        """
        out = {
            "blended_rates": cur_data.get("blended_rates", {}),
            "usage_summary": cur_data.get("usage_summary", {}),
            "status": cur_data.get("status", "unknown"),
            "fetch_timestamp": cur_data.get("fetch_timestamp", ""),
            "period_days": cur_data.get("period_days", 30),
        }

        if cur_data.get("status") == "failed":
            logger.warning(
                "DataNormalizer: CUR fetch failed — aws_actual_costs.json written with empty "
                "blended_rates. This is normal until Cost Explorer is enabled and "
                "ce:GetCostAndUsage IAM permission is added."
            )

        anomalies = [r for r, s in out["usage_summary"].items() if s.get("anomaly", False)]
        if anomalies:
            logger.warning("Cost anomalies detected in: %s", anomalies)

        self._atomic_write(self._actual_costs_path, out)
        logger.info("DataNormalizer: wrote actual costs → %s", self._actual_costs_path)
        return out

    # -----------------------------------------------------------------------
    # Public: carbon
    # -----------------------------------------------------------------------

    def normalize_carbon(self, raw_carbon: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Normalises carbon data and writes to data/carbon/carbon_intensity.json.
        Ensures both gco2_per_kwh and carbon_intensity_gco2_per_kwh keys exist.
        """
        out: Dict[str, Dict] = {}

        for region, data in raw_carbon.items():
            ci = (
                data.get("carbon_intensity_gco2_per_kwh")
                or data.get("gco2_per_kwh")
                or 400.0
            )
            out[region] = {
                **data,
                "carbon_intensity_gco2_per_kwh": round(float(ci), 2),
                "gco2_per_kwh": round(float(ci), 2),
            }

        self._atomic_write(self._carbon_path, out)

        live = sum(1 for v in out.values() if "live" in v.get("data_source", ""))
        stat = sum(1 for v in out.values() if "static" in v.get("data_source", ""))
        logger.info(
            "DataNormalizer: wrote carbon → %s (%d live, %d static)",
            self._carbon_path,
            live,
            stat,
        )
        return out

    # -----------------------------------------------------------------------
    # Public: convenience reader
    # -----------------------------------------------------------------------

    def get_flat_carbon(self) -> Dict[str, float]:
        """
        Reads carbon JSON and returns {region: gco2_per_kwh} flat dict.
        Compatible with StateBuilder.build() carbon parameter.
        Falls back to static values if file not found or empty.
        """
        try:
            with open(self._carbon_path, encoding="utf-8") as fh:
                data = json.load(fh)
            if data:
                return {
                    region: float(entry.get("gco2_per_kwh", 400.0))
                    for region, entry in data.items()
                }
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            pass

        from ai_engine.data_pipeline.carbon_api_client import STATIC_CARBON_INTENSITY

        logger.debug("DataNormalizer.get_flat_carbon: using static fallback.")
        return {r: v["gco2_kwh"] for r, v in STATIC_CARBON_INTENSITY.items()}

    # -----------------------------------------------------------------------
    # Compatibility methods expected by test_data_pipeline.py
    # -----------------------------------------------------------------------

    def write_pricing(self, data: dict) -> None:
        """
        Write pricing data to the configured output path.
        If data is empty, writes static fallback values instead of {}.
        Uses atomic write (tmp → rename) to avoid partial reads.
        """
        output_path = Path(
            self._config.get("data_pipeline", {}).get(
                "pricing_output_path", "data/pricing/aws_pricing.json"
            )
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not data:
            data = self._static_pricing_fallback()

        tmp_path = output_path.with_suffix(".tmp.json")
        try:
            tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            tmp_path.replace(output_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def write_carbon(self, data: dict) -> None:
        """
        Write carbon intensity data to the configured output path.
        Uses atomic write (tmp → rename).
        """
        output_path = Path(
            self._config.get("data_pipeline", {}).get(
                "carbon_output_path", "data/carbon/carbon_intensity.json"
            )
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not data:
            data = self._static_carbon_fallback()

        tmp_path = output_path.with_suffix(".tmp.json")
        try:
            tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            tmp_path.replace(output_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    @staticmethod
    def _static_pricing_fallback() -> dict:
        return {
            "us-east-1": {"on_demand_per_vcpu_hr": 0.096},
            "us-west-2": {"on_demand_per_vcpu_hr": 0.096},
            "eu-west-1": {"on_demand_per_vcpu_hr": 0.107},
            "eu-central-1": {"on_demand_per_vcpu_hr": 0.111},
            "eu-north-1": {"on_demand_per_vcpu_hr": 0.098},
            "ap-southeast-1": {"on_demand_per_vcpu_hr": 0.114},
            "ap-northeast-1": {"on_demand_per_vcpu_hr": 0.118},
            "ca-central-1": {"on_demand_per_vcpu_hr": 0.100},
        }

    @staticmethod
    def _static_carbon_fallback() -> dict:
        return {
            "us-east-1": {"gco2_per_kwh": 415.0},
            "us-west-2": {"gco2_per_kwh": 192.0},
            "eu-west-1": {"gco2_per_kwh": 316.0},
            "eu-central-1": {"gco2_per_kwh": 338.0},
            "eu-north-1": {"gco2_per_kwh": 42.0},
            "ap-southeast-1": {"gco2_per_kwh": 453.0},
        }

    # -----------------------------------------------------------------------
    # Private: atomic write
    # -----------------------------------------------------------------------

    @staticmethod
    def _atomic_write(path: Path, data: Dict):
        """
        Writes JSON to .tmp.json then renames to final path.
        Readers never see a partial file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.json")
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, default=str)
            tmp.replace(path)
        except Exception as exc:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            raise exc