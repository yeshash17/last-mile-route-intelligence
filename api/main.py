"""
api/main.py — FastAPI decision server.

Endpoints:
  POST /optimize-route  — full pipeline: OSRM + service_time_model + failure_predictor + PyVRP
  POST /risk-score      — score a single stop's failure probability
  GET  /health          — liveness check + model status

Run:
  uvicorn api.main:app --reload --port 8000
  http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional
import logging
import numpy as np

from api.schemas import (
    OptimiseRouteRequest, OptimiseRouteResponse,
    RiskScoreRequest, RiskResult, RouteStepResponse, DeferredStop,
    HealthResponse,
)
from optimizer.vrp_solver import solve_vrp, Stop as VRPStop
from models.failure_predictor import FailurePredictor
from models.service_time import ServiceTimeModel
from models.driver_profile import get_driver_profile
from data.distance_matrix import build_matrix
from config import MODEL_DIR

logger = logging.getLogger(__name__)

_BUILDING_TYPE_MAP = {
    "house": 1, "flat": 1, "business": 2, "locker": 3, "other": 0
}

# ── Model registry ────────────────────────────────────────────────────────────

models: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading ML models...")

    try:
        models["failure"] = FailurePredictor.load(MODEL_DIR)
        logger.info("Failure predictor loaded.")
    except FileNotFoundError:
        logger.warning("Failure predictor not found — run models/trainer.py first.")
        models["failure"] = None

    try:
        models["service_time"] = ServiceTimeModel.load(MODEL_DIR)
        logger.info("Service time model loaded (%d AOIs).", len(models["service_time"].aoi_stats))
    except FileNotFoundError:
        logger.warning("Service time model not found — run models/service_time.py first.")
        models["service_time"] = None

    yield
    models.clear()


app = FastAPI(
    title       = "Route Intelligence API",
    description = "Last-mile delivery decision engine — OSRM + service time model + PyVRP VRPTW.",
    version     = "0.2.0",
    lifespan    = lifespan,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dwell_p75(stop) -> float:
    """Service time at P75 from learned model, or global mean * 1.15 if unseen."""
    svc = models.get("service_time")
    if svc:
        aoi_type = _BUILDING_TYPE_MAP.get(stop.building_type, 0)
        p50 = svc.predict(
            aoi_id        = -1,
            aoi_type      = aoi_type,
            package_count = max(1, round(stop.package_weight_kg)),
            hour_of_day   = (stop.planned_arrival_dt or datetime.now()).hour,
        )
        return round(p50 * 1.15, 1)   # P75 ≈ mean * 1.15 for right-skewed distribution
    return 10.0   # fallback global mean if model not loaded


def _p_success(stop, hour: int) -> float:
    """P(delivery succeeds). Uses failure predictor if loaded, else historical rate."""
    fail_model = models.get("failure")
    if fail_model:
        feats = {
            "hour_of_day":               hour,
            "day_of_week":               datetime.now().weekday(),
            "is_weekend":                int(datetime.now().weekday() >= 5),
            "is_peak_hour":              int(hour in (8, 9, 17, 18)),
            "building_type":             _BUILDING_TYPE_MAP.get(stop.building_type, 0),
            "has_access_code":           int(stop.has_access_code),
            "has_safe_place":            int(stop.has_safe_place),
            "package_weight_kg":         stop.package_weight_kg,
            "is_heavy":                  int(stop.package_weight_kg > 5),
            "n_previous_failed_attempts": stop.n_previous_failed_attempts,
            "historical_success_rate":   stop.historical_success_rate,
            "hour_success_rate":         stop.historical_success_rate,
            "avg_dwell_time_mins":       _dwell_p75(stop),
            "n_past_deliveries":         10,
        }
        result = fail_model.predict(feats)
        return round(1.0 - result["failure_probability"], 3)
    return stop.historical_success_rate


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/optimize-route", response_model=OptimiseRouteResponse)
async def optimize_route(req: OptimiseRouteRequest):
    """
    Full pipeline:
      1. Build OSRM time matrix (real road times)
      2. Score dwell time per stop from service_time_model (P75)
      3. Score P(success) per stop from failure_predictor
      4. Solve VRPTW with PyVRP (zone-coherent, FADR-aware, multi-objective)
      5. Return ordered route with ETAs, risk scores, deferred stops
    """
    if not req.stops:
        raise HTTPException(status_code=422, detail="Stop list is empty.")

    dispatch_hour = req.shift_start.hour

    # ── Step 1: time matrix (depot + all stops) ───────────────────────────────
    coords = [(req.depot_lat, req.depot_lon)] + [(s.lat, s.lon) for s in req.stops]
    try:
        matrix = np.array(build_matrix(coords), dtype=float)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Distance matrix build failed: {e}")

    # ── Step 2 & 3: score each stop ───────────────────────────────────────────
    vrp_stops = []
    risk_by_id: dict = {}

    for s in req.stops:
        hour      = dispatch_hour + max(0, s.time_window_open_mins // 60)
        dwell     = _dwell_p75(s)
        p_ok      = _p_success(s, hour)
        aoi_type  = _BUILDING_TYPE_MAP.get(s.building_type, 0)

        # Full risk result for response
        fail_model = models.get("failure")
        if fail_model:
            feats = {
                "hour_of_day": hour, "day_of_week": datetime.now().weekday(),
                "is_weekend": int(datetime.now().weekday() >= 5),
                "is_peak_hour": int(hour in (8, 9, 17, 18)),
                "building_type": aoi_type,
                "has_access_code": int(s.has_access_code),
                "has_safe_place": int(s.has_safe_place),
                "package_weight_kg": s.package_weight_kg,
                "is_heavy": int(s.package_weight_kg > 5),
                "n_previous_failed_attempts": s.n_previous_failed_attempts,
                "historical_success_rate": s.historical_success_rate,
                "hour_success_rate": s.historical_success_rate,
                "avg_dwell_time_mins": dwell,
                "n_past_deliveries": 10,
            }
            risk_by_id[s.stop_id] = RiskResult(**fail_model.predict(feats))

        vrp_stops.append(VRPStop(
            stop_id           = s.stop_id,
            address           = s.address,
            lat               = s.lat,
            lon               = s.lon,
            time_window_open  = s.time_window_open_mins,
            time_window_close = s.time_window_close_mins,
            demand_kg         = s.package_weight_kg,
            dwell_mins        = dwell,
            p_success         = p_ok,
            aoi_type          = aoi_type,
            zone_id           = s.zone_id,
        ))

    # ── Step 3b: driver familiarity — adjust dwell before solver sees it ───────
    driver_profile = get_driver_profile(req.driver_id, MODEL_DIR)
    if driver_profile.familiar_aois:
        for vs in vrp_stops:
            vs.dwell_mins = driver_profile.adjusted_dwell(vs)

    # ── Step 4: solve ─────────────────────────────────────────────────────────
    result = solve_vrp(
        stops               = vrp_stops,
        time_matrix         = matrix,
        num_vehicles        = 1,
        vehicle_capacity_kg = req.vehicle_capacity_kg,
        shift_duration_mins = req.shift_duration_mins,
        apply_zone_penalty  = True,
        deferral_enabled    = True,
        driver_profile      = driver_profile,
    )

    if result is None:
        raise HTTPException(
            status_code = 422,
            detail      = "No feasible route found. Check time windows fit within shift duration "
                          "and total weight is within vehicle capacity.",
        )

    # ── Step 5: build response ────────────────────────────────────────────────
    route_steps = []
    for route in result.routes:
        for step in route.steps:
            route_steps.append(RouteStepResponse(
                sequence    = step.sequence,
                stop_id     = step.stop_id,
                address     = step.address,
                arrival_min = step.arrival_min,
                dwell_mins  = step.dwell_mins,
                p_success   = step.p_success,
                risk        = risk_by_id.get(step.stop_id),
            ))

    deferred = [
        DeferredStop(
            stop_id   = s.stop_id,
            address   = s.address,
            p_success = s.p_success,
        )
        for s in result.deferred
    ]

    return OptimiseRouteResponse(
        driver_id       = req.driver_id,
        total_stops     = len(route_steps),
        total_time_mins = result.total_time,
        expected_fadr   = result.expected_fadr,
        cascade_risk    = result.cascade_risk,
        route           = route_steps,
        deferred_stops  = deferred,
        cost_breakdown  = result.cost_breakdown,
        generated_at    = datetime.utcnow(),
    )


@app.post("/risk-score", response_model=RiskResult)
async def risk_score(req: RiskScoreRequest):
    """Score a single stop's failure probability for dispatcher triage."""
    if not models.get("failure"):
        raise HTTPException(status_code=503,
                            detail="Failure predictor not loaded. Run models/trainer.py first.")
    feats = {
        "hour_of_day":               req.planned_hour,
        "day_of_week":               req.planned_day_of_week,
        "is_weekend":                int(req.planned_day_of_week >= 5),
        "is_peak_hour":              int(req.planned_hour in (8, 9, 17, 18)),
        "building_type":             _BUILDING_TYPE_MAP.get(req.building_type, 0),
        "has_access_code":           int(req.has_access_code),
        "has_safe_place":            int(req.has_safe_place),
        "package_weight_kg":         req.package_weight_kg,
        "is_heavy":                  int(req.package_weight_kg > 5),
        "n_previous_failed_attempts": req.n_previous_failed_attempts,
        "historical_success_rate":   req.historical_success_rate,
        "hour_success_rate":         req.historical_success_rate,
        "avg_dwell_time_mins":       5.0,
        "n_past_deliveries":         10,
    }
    return RiskResult(**models["failure"].predict(feats))


@app.get("/health", response_model=HealthResponse)
async def health():
    svc = models.get("service_time")
    return HealthResponse(
        status = "ok",
        models = {
            "failure_predictor": "loaded" if models.get("failure") else "not loaded",
            "service_time_model": (
                f"loaded ({len(svc.aoi_stats):,} AOIs)" if svc else "not loaded"
            ),
        },
    )
