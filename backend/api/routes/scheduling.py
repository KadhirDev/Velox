"""
Scheduling API Routes
======================

POST /api/v1/schedule       — submit workload, get RL placement decision
GET  /api/v1/decisions      — list recent decisions
GET  /api/v1/decisions/{id} — get single decision
POST /api/v1/decisions/{id}/explain — compute SHAP explanation in background
POST /api/v1/batch          — submit multiple workloads
GET  /api/v1/status         — agent/system status
"""

import asyncio
import logging
import os
import time
import uuid
from typing import Any, Dict, Optional

import numpy as np
import yaml
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool

from ai_engine.operator.operator import VeloxOperator
from backend.api.models.schemas import (
    AgentStatusResponse,
    BatchSchedulingResponse,
    BatchWorkloadRequest,
    DecisionListResponse,
    SchedulingDecision,
    WorkloadRequest,
)
from backend.auth.security import can_schedule
from backend.core.agent_singleton import get_agent, get_producer
from backend.core.decision_store import DecisionStore

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["scheduling"])

# In-memory store
_decision_store = DecisionStore(max_size=1000)

# Integer -> label mapping for model/action output
SLA_TIER_MAP = {
    0: "best_effort",
    1: "bronze",
    2: "silver",
    3: "gold",
    4: "platinum",
    5: "critical",
}

# -----------------------------------------------------------------------------
# Inference concurrency gate
# -----------------------------------------------------------------------------
# Limit concurrent blocking inference before entering the thread pool.
# This should track the safer CPU-aware pool sizing strategy.
_CPU_COUNT = os.cpu_count() or 4
_MAX_CONCURRENT_INFERENCE = min(32, max(8, _CPU_COUNT * 3))
_infer_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_INFERENCE)

logger.info(
    "SchedulingRouter: inference semaphore set to %d (%d CPUs detected)",
    _MAX_CONCURRENT_INFERENCE,
    _CPU_COUNT,
)


# =============================================================================
# Internal helpers
# =============================================================================

def _load_config() -> dict:
    """Load application config safely."""
    try:
        with open("config/settings.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.warning("config/settings.yaml not found; using empty config")
        return {}
    except Exception as exc:
        logger.warning("Failed to load config/settings.yaml: %s", exc)
        return {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert value to float safely."""
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_str(value: Any, default: str = "") -> str:
    """Convert value to string safely."""
    if value is None:
        return default
    return str(value)


def _normalize_sla_tier(value: Any) -> str:
    """Normalize SLA tier from int/model output into API string."""
    if value is None:
        return "standard"
    if isinstance(value, int):
        return SLA_TIER_MAP.get(value, "standard")
    if isinstance(value, float) and value.is_integer():
        return SLA_TIER_MAP.get(int(value), "standard")
    return str(value)


def _extract_internals(raw: Dict[str, Any]) -> tuple[Dict[str, Any], np.ndarray, Dict[str, Any]]:
    """
    Extract internal agent fields before schema conversion.

    Returns:
        cleaned_raw, state, decoded
    """
    raw = dict(raw or {})
    state = raw.pop("_state", np.zeros(45, dtype=np.float32))
    decoded = raw.pop("_decoded", {})

    try:
        state = np.asarray(state, dtype=np.float32)
    except Exception:
        state = np.zeros(45, dtype=np.float32)

    if not isinstance(decoded, dict):
        try:
            if hasattr(decoded, "model_dump"):
                decoded = decoded.model_dump()
            elif hasattr(decoded, "dict"):
                decoded = decoded.dict()
            elif hasattr(decoded, "__dict__"):
                decoded = {k: v for k, v in vars(decoded).items() if not k.startswith("_")}
            else:
                decoded = {}
        except Exception:
            decoded = {}

    return raw, state, decoded


def _to_scheduling_decision(
    raw: Dict[str, Any],
    *,
    workload_id: str,
    decision_id: str,
    latency_ms: float,
) -> SchedulingDecision:
    """
    Convert raw scheduler/operator output into the API response model safely.
    """
    raw = raw or {}

    return SchedulingDecision(
        decision_id=_safe_str(raw.get("decision_id", decision_id), decision_id),
        workload_id=_safe_str(raw.get("workload_id", workload_id), workload_id),
        cloud=_safe_str(raw.get("cloud", "unknown"), "unknown"),
        region=_safe_str(raw.get("region", "unknown"), "unknown"),
        instance_type=_safe_str(raw.get("instance_type", "unknown"), "unknown"),
        scaling_action=_safe_str(raw.get("scaling_action", "none"), "none"),
        purchase_option=_safe_str(raw.get("purchase_option", "on_demand"), "on_demand"),
        sla_tier=_normalize_sla_tier(raw.get("sla_tier", "standard")),
        estimated_cost_per_hr=_safe_float(raw.get("estimated_cost_per_hr", 0.0), 0.0),
        cost_savings_pct=_safe_float(raw.get("cost_savings_pct", 0.0), 0.0),
        carbon_savings_pct=_safe_float(raw.get("carbon_savings_pct", 0.0), 0.0),
        latency_ms=_safe_float(raw.get("latency_ms", latency_ms), latency_ms),
        explanation=raw.get("explanation") or {},
        actual_reward=raw.get("actual_reward"),
    )


def _heuristic_fallback_decision(request: WorkloadRequest) -> SchedulingDecision:
    """
    Use VeloxOperator heuristic decision path when the RL agent
    is still loading, instead of returning HTTP 503.
    """
    config = _load_config()

    op = VeloxOperator(
        config=config,
        dry_run=True,
        no_kafka=True,
        no_shap=True,
    )
    op._agent = None

    workload = request.to_agent_dict()

    fallback_workload_id = (
        getattr(request, "workload_id", None)
        or workload.get("workload_id")
        or "api-fallback"
    )
    fallback_decision_id = str(uuid.uuid4())

    workload["workload_id"] = fallback_workload_id

    raw = op._heuristic_decision(workload) or {}
    raw["decision_id"] = fallback_decision_id
    raw["workload_id"] = fallback_workload_id
    raw["latency_ms"] = 1.0
    raw.setdefault("explanation", {})
    raw.setdefault("sla_tier", "standard")

    return _to_scheduling_decision(
        raw,
        workload_id=fallback_workload_id,
        decision_id=fallback_decision_id,
        latency_ms=1.0,
    )


async def _run_agent_inference(agent, workload: Dict[str, Any]):
    """
    Run blocking model inference in FastAPI's thread pool while gating
    concurrency with an async semaphore to reduce CPU oversubscription.
    """
    async with _infer_semaphore:
        try:
            if hasattr(agent, "decide"):
                try:
                    return await run_in_threadpool(agent.decide, workload, False)
                except TypeError:
                    return await run_in_threadpool(agent.decide, workload)
            if hasattr(agent, "schedule"):
                try:
                    return await run_in_threadpool(agent.schedule, workload, False)
                except TypeError:
                    return await run_in_threadpool(agent.schedule, workload)
            raise RuntimeError("Agent has neither decide() nor schedule()")
        except Exception as exc:
            logger.error("SchedulerAgent inference failed: %s", exc, exc_info=True)
            raise


# =============================================================================
# POST /api/v1/schedule
# =============================================================================

@router.post(
    "/schedule",
    response_model=SchedulingDecision,
    status_code=status.HTTP_200_OK,
    summary="Submit a workload for RL scheduling",
)
async def schedule_workload(
    request: WorkloadRequest,
    background_tasks: BackgroundTasks,
    agent=Depends(get_agent),
    producer=Depends(get_producer),
    _auth=Depends(can_schedule),
) -> SchedulingDecision:
    if agent is None:
        decision = _heuristic_fallback_decision(request)
        background_tasks.add_task(
            _store_and_publish,
            producer,
            decision,
            request.to_agent_dict(),
            np.zeros(45, dtype=np.float32),
            {},
        )
        return decision

    t0 = time.perf_counter()
    decision_id = str(uuid.uuid4())

    workload = request.to_agent_dict()
    workload_id = getattr(request, "workload_id", None) or decision_id
    workload["workload_id"] = workload_id

    try:
        raw = await _run_agent_inference(agent, workload)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RL inference error: {str(exc)[:200]}",
        )

    if raw is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent returned no decision. Model may be uninitialised.",
        )

    total_latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    raw, state, decoded = _extract_internals(raw)
    raw["decision_id"] = decision_id
    raw["workload_id"] = workload_id
    raw["latency_ms"] = total_latency_ms
    raw.setdefault("explanation", {})
    raw.setdefault("sla_tier", "standard")

    try:
        decision = _to_scheduling_decision(
            raw,
            workload_id=workload_id,
            decision_id=decision_id,
            latency_ms=total_latency_ms,
        )
    except Exception as exc:
        logger.error("SchedulingDecision conversion failed: %s | raw=%s", exc, raw, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Response schema error: {str(exc)[:300]}",
        )

    background_tasks.add_task(
        _store_and_publish,
        producer,
        decision,
        workload,
        state,
        decoded,
    )

    logger.info(
        "Decision %s: %s/%s %s cost=%.4f/hr savings=%.1f%% %.0fms",
        decision.decision_id[:8],
        decision.cloud,
        decision.region,
        decision.purchase_option,
        decision.estimated_cost_per_hr,
        decision.cost_savings_pct,
        decision.latency_ms,
    )

    return decision


# =============================================================================
# POST /api/v1/decisions/{decision_id}/explain
# =============================================================================

@router.post(
    "/decisions/{decision_id}/explain",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Compute SHAP explanation for a stored decision",
)
async def explain_decision(
    decision_id: str,
    background_tasks: BackgroundTasks,
    agent=Depends(get_agent),
):
    record = _decision_store.get_record(decision_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Decision '{decision_id}' not found. "
                f"Decisions persist only for the lifetime of this pod."
            ),
        )

    if agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RL agent not loaded.",
        )

    explainer = getattr(agent, "_explainer", None)
    if explainer is None:
        attempted = getattr(agent, "_shap_init_attempted", False)
        detail = (
            "SHAP explainer initialised but failed to load. Check logs for SHAPExplainer errors."
            if attempted
            else "SHAP explainer still initialising. Retry in 30 seconds."
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
        )

    existing = getattr(record.decision, "explanation", None)
    if (
        isinstance(existing, dict)
        and not existing.get("error")
        and (existing.get("summary") or "top_drivers" in existing)
    ):
        return {
            "status": "already_complete",
            "decision_id": decision_id,
            "message": (
                "Explanation already available. "
                "Fetch via GET /api/v1/decisions/{decision_id}."
            ),
        }

    if getattr(record, "_explain_in_progress", False):
        return {
            "status": "in_progress",
            "decision_id": decision_id,
            "message": "Explanation already computing.",
        }

    try:
        record._explain_in_progress = True
    except Exception:
        pass

    background_tasks.add_task(
        _compute_and_attach_explanation,
        agent,
        decision_id,
        record.state,
        record.decoded,
    )

    return {
        "status": "accepted",
        "decision_id": decision_id,
        "message": (
            "SHAP computing in background. "
            "Poll GET /api/v1/decisions/{decision_id} in ~10–15s."
        ),
    }


# =============================================================================
# POST /api/v1/batch
# =============================================================================

@router.post(
    "/batch",
    response_model=BatchSchedulingResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit multiple workloads for batch scheduling",
)
async def schedule_batch(
    request: BatchWorkloadRequest,
    background_tasks: BackgroundTasks,
    agent=Depends(get_agent),
    producer=Depends(get_producer),
    _auth=Depends(can_schedule),
) -> BatchSchedulingResponse:
    if len(request.workloads) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 workloads per batch")

    decisions = []
    errors = []
    t0 = time.perf_counter()

    async def _infer_one(i: int, wl_req: WorkloadRequest):
        try:
            if agent is None:
                d = _heuristic_fallback_decision(wl_req)
                background_tasks.add_task(
                    _store_and_publish,
                    producer,
                    d,
                    wl_req.to_agent_dict(),
                    np.zeros(45, dtype=np.float32),
                    {},
                )
                return d, None

            item_t0 = time.perf_counter()
            workload = wl_req.to_agent_dict()
            decision_id = str(uuid.uuid4())
            workload_id = getattr(wl_req, "workload_id", None) or f"batch-{i}"
            workload["workload_id"] = workload_id

            raw = await _run_agent_inference(agent, workload)

            if raw is None:
                return None, {
                    "index": i,
                    "error": "Agent returned no decision. Model may be uninitialised.",
                }

            item_latency_ms = round((time.perf_counter() - item_t0) * 1000, 2)

            raw, state, decoded = _extract_internals(raw)
            raw["decision_id"] = decision_id
            raw["workload_id"] = workload_id
            raw["latency_ms"] = item_latency_ms
            raw.setdefault("explanation", {})
            raw.setdefault("sla_tier", "standard")

            d = _to_scheduling_decision(
                raw,
                workload_id=workload_id,
                decision_id=decision_id,
                latency_ms=item_latency_ms,
            )

            background_tasks.add_task(
                _store_and_publish,
                producer,
                d,
                workload,
                state,
                decoded,
            )
            return d, None

        except Exception as exc:
            logger.error("Batch scheduling failed for index %s: %s", i, exc, exc_info=True)
            return None, {"index": i, "error": str(exc)[:200]}

    results = await asyncio.gather(
        *[_infer_one(i, wl_req) for i, wl_req in enumerate(request.workloads)]
    )

    for d, err in results:
        if d is not None:
            decisions.append(d)
        if err is not None:
            errors.append(err)

    return BatchSchedulingResponse(
        decisions=decisions,
        errors=errors,
        total_latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        count=len(decisions),
    )


# =============================================================================
# GET /api/v1/decisions
# =============================================================================

@router.get(
    "/decisions",
    response_model=DecisionListResponse,
    summary="List recent scheduling decisions",
)
async def list_decisions(
    limit: int = 20,
    cloud: Optional[str] = None,
    region: Optional[str] = None,
) -> DecisionListResponse:
    decisions = _decision_store.list(limit=limit, cloud=cloud, region=region)
    return DecisionListResponse(decisions=decisions, count=len(decisions))


# =============================================================================
# GET /api/v1/decisions/{decision_id}
# =============================================================================

@router.get(
    "/decisions/{decision_id}",
    response_model=SchedulingDecision,
    summary="Get a single decision",
)
async def get_decision(decision_id: str) -> SchedulingDecision:
    d = _decision_store.get(decision_id)
    if d is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Decision {decision_id} not found. "
                f"Decisions are kept in memory for the lifetime of this pod."
            ),
        )
    return d


# =============================================================================
# GET /api/v1/status
# =============================================================================

@router.get(
    "/status",
    response_model=AgentStatusResponse,
    summary="Agent and system status",
)
async def agent_status(agent=Depends(get_agent)) -> AgentStatusResponse:
    last = _decision_store.last()

    explainer = getattr(agent, "_explainer", None) if agent else None
    config = getattr(agent, "_config", {}) if agent else {}
    model_cfg = config.get("model", {}) if isinstance(config, dict) else {}

    background_shape = []
    if explainer is not None:
        get_bg_shape = getattr(explainer, "get_background_shape", None)
        if callable(get_bg_shape):
            try:
                background_shape = list(get_bg_shape())
            except Exception:
                background_shape = []

    return AgentStatusResponse(
        agent_loaded=agent is not None,
        model_path=str(model_cfg.get("path", "")) if agent else "",
        shap_ready=explainer is not None,
        background_shape=background_shape,
        decisions_served=_decision_store.total_count(),
        last_decision_id=last.decision_id if last else None,
        last_decision_cloud=last.cloud if last else None,
        last_decision_region=last.region if last else None,
    )


# =============================================================================
# Background task helpers
# =============================================================================

def _store_and_publish(producer, decision, workload, state, decoded):
    """
    Store first, then attempt Kafka publish.
    Runs inside FastAPI BackgroundTasks threadpool.
    Never raises.
    """
    try:
        _decision_store.put(decision, workload, state, decoded)
    except TypeError:
        try:
            _decision_store.put(decision)
        except Exception as exc:
            logger.warning("DecisionStore.put failed (non-fatal): %s", exc)
    except Exception as exc:
        logger.warning("DecisionStore.put failed (non-fatal): %s", exc)

    if producer is None:
        return

    try:
        producer.publish_decision(
            {
                "decision_id": decision.decision_id,
                "workload_id": decision.workload_id,
                "cloud": decision.cloud,
                "region": decision.region,
                "instance_type": decision.instance_type,
                "purchase_option": decision.purchase_option,
                "cost_savings_pct": decision.cost_savings_pct,
                "carbon_savings_pct": decision.carbon_savings_pct,
                "latency_ms": decision.latency_ms,
                "estimated_cost_per_hr": decision.estimated_cost_per_hr,
                "workload_type": (
                    workload.get("workload_type", "batch")
                    if isinstance(workload, dict)
                    else "batch"
                ),
                "explanation": decision.explanation or {},
                "actual_reward": getattr(decision, "actual_reward", None),
            }
        )
    except Exception as exc:
        logger.warning("Kafka publish failed (non-fatal): %s", exc)


def _compute_and_attach_explanation(agent, decision_id, state, decoded):
    """
    Runs in BackgroundTasks thread pool.
    Guarantees: explanation is attached if store update succeeds.
    Never raises.
    """
    t0 = time.perf_counter()
    logger.info("SHAP: starting computation for decision %s", decision_id[:8])

    explanation = None
    try:
        if hasattr(agent, "compute_explanation"):
            explanation = agent.compute_explanation(state, decoded)
        elif hasattr(agent, "_build_explanation"):
            explanation = agent._build_explanation(
                raw_state=state,
                norm_state=state,
                action=[],
                decoded=decoded,
                workload=decoded,
                workload_dict=decoded if isinstance(decoded, dict) else {},
            )
    except Exception as exc:
        logger.error(
            "SHAP: compute_explanation raised for %s: %s",
            decision_id[:8],
            exc,
            exc_info=True,
        )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    has_summary = bool(
        explanation
        and isinstance(explanation, dict)
        and explanation.get("summary")
    )
    has_drivers_key = bool(
        explanation
        and isinstance(explanation, dict)
        and "top_drivers" in explanation
    )

    if not has_summary or not has_drivers_key:
        logger.warning(
            "SHAP: output incomplete for %s "
            "(has_summary=%s has_drivers=%s elapsed=%.0fms) — using fallback",
            decision_id[:8],
            has_summary,
            has_drivers_key,
            elapsed_ms,
        )
        explanation = {
            "summary": (
                "SHAP analysis completed but produced no clear attribution signal. "
                "This is normal for early-stage or undertrained RL models."
            ),
            "top_drivers": [],
            "top_positive": [],
            "top_negative": [],
            "base_value": 0.0,
            "confidence": 0.0,
            "explanation_ms": round(elapsed_ms, 1),
            "error": False,
        }

    attached = _decision_store.attach_explanation(decision_id, explanation)

    record = _decision_store.get_record(decision_id)
    if record is not None:
        try:
            record._explain_in_progress = False
        except Exception:
            pass

    logger.info(
        "SHAP: complete for %s — attached=%s confidence=%.3f elapsed=%.0fms",
        decision_id[:8],
        attached,
        float(explanation.get("confidence", 0.0) or 0.0),
        elapsed_ms,
    )

    if not attached:
        logger.error(
            "SHAP: CRITICAL — attach_explanation failed for %s. "
            "Decision will show no explanation in UI.",
            decision_id[:8],
        )