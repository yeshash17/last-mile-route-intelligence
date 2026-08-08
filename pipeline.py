"""
pipeline.py

Full end-to-end integration test:
  OSRM (real road times) -> service_time_model (P75 dwell) -> failure_predictor (P_success)
  -> vrp_solver (PyVRP VRPTW) -> route output

Run: python pipeline.py
Requires:
  - OSRM running on Docker (docker ps to verify osrm-uk container)
  - saved_models/ populated (run models/service_time.py first)
  - saved_models/failure_predictor.joblib (run models/trainer.py first)
"""

import sys
import logging
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent))
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from data.distance_matrix import build_matrix
from models.service_time import ServiceTimeModel
from models.failure_predictor import FailurePredictor, FEATURES
from optimizer.vrp_solver import Stop, solve_vrp

# ── Test scenario: London delivery route ──────────────────────────────────────
# Depot: Euston station
# 8 stops across central London — mix of zones, building types, risk levels
# OSRM has Great Britain OSM data so real road times apply

DEPOT = (51.5282, -0.1337)   # Euston Station

STOPS_RAW = [
    # (lat, lon, address, zone_id, aoi_type, demand_kg, tw_open, tw_close,
    #  building_type, has_access_code, has_safe_place, hour_success_rate,
    #  historical_success_rate, n_past_deliveries, n_previous_failed)
    (51.5238, -0.1585, "Baker St, NW1",        "A", 1, 3.0,  60, 300,  1, 0, 0, 0.88, 0.91, 12, 0),
    (51.5155, -0.1416, "Oxford Circus, W1",     "A", 2, 5.0,  90, 360,  2, 1, 0, 0.82, 0.88, 20, 1),
    (51.5142, -0.1494, "Bond St, W1S",          "A", 2, 2.0,   0, 480,  2, 0, 1, 0.90, 0.93,  8, 0),
    (51.5074, -0.1278, "Trafalgar Sq, WC2",     "B", 1, 4.0, 120, 420,  1, 0, 0, 0.85, 0.87, 15, 0),
    (51.4994, -0.1273, "Victoria, SW1",         "B", 1, 3.5, 180, 420,  1, 1, 0, 0.79, 0.83, 30, 2),
    (51.5033, -0.1195, "Waterloo, SE1",         "C", 2, 6.0,  60, 360,  2, 0, 0, 0.91, 0.94,  5, 0),
    (51.5114, -0.1198, "Holborn, WC1",          "C", 2, 2.5,   0, 480,  2, 1, 1, 0.87, 0.90, 22, 0),
    (51.5196, -0.0754, "Shoreditch, E1",        "D", 1, 4.0,   0, 480,  1, 0, 0, 0.35, 0.40,  3, 3),  # high-risk
]

SHIFT_DURATION_MINS  = 480   # 8-hour shift
NUM_VEHICLES         = 3
DISPATCH_HOUR        = 9     # 09:00


def _make_stop_features(row: tuple, hour: int, dwell_hist: float) -> dict:
    """Build feature dict for failure_predictor from raw stop data."""
    _, _, _, _, aoi_type, demand_kg, *_, building_type, has_access, has_safe, \
        hour_rate, hist_rate, n_past, n_failed = row
    return {
        "hour_of_day":              hour,
        "day_of_week":              datetime.now().weekday(),
        "is_weekend":               int(datetime.now().weekday() >= 5),
        "is_peak_hour":             int(hour in (8, 9, 17, 18)),
        "building_type":            building_type,
        "has_access_code":          has_access,
        "has_safe_place":           has_safe,
        "package_weight_kg":        demand_kg,
        "is_heavy":                 int(demand_kg > 5),
        "n_previous_failed_attempts": n_failed,
        "historical_success_rate":  hist_rate,
        "hour_success_rate":        hour_rate,
        "avg_dwell_time_mins":      dwell_hist,
        "n_past_deliveries":        n_past,
    }


def run_pipeline():
    print("\n" + "=" * 65)
    print("LAST-MILE ROUTE INTELLIGENCE ENGINE  — full pipeline test")
    print(f"Scenario: {len(STOPS_RAW)} London stops | {NUM_VEHICLES} vehicles | {SHIFT_DURATION_MINS}min shift")
    print("=" * 65)

    # ── Step 1: load models ───────────────────────────────────────────────────
    print("\n[1/4] Loading models...")
    try:
        svc_model = ServiceTimeModel.load()
        print(f"  service_time_model  loaded  (global mean={svc_model.global_mean:.1f} min, "
              f"{len(svc_model.aoi_stats):,} AOIs learned)")
    except Exception as e:
        print(f"  service_time_model  FAILED: {e}")
        print("  Run: python -m models.service_time  first")
        svc_model = None

    try:
        fail_model = FailurePredictor.load()
        print(f"  failure_predictor   loaded")
    except Exception as e:
        print(f"  failure_predictor   not available ({e}) — using historical_success_rate as P(success)")
        fail_model = None

    # ── Step 2: build stop objects with model predictions ────────────────────
    print("\n[2/4] Scoring stops...")
    coords = [DEPOT] + [(r[0], r[1]) for r in STOPS_RAW]
    stops  = []

    for i, row in enumerate(STOPS_RAW):
        lat, lon, address, zone_id, aoi_type, demand_kg, tw_open, tw_close = row[:8]

        # Service time from model (P75 = mean + 0.674 * std ≈ mean * 1.15 rough)
        if svc_model:
            p50 = svc_model.predict(aoi_id=-1, aoi_type=aoi_type,
                                    package_count=max(1, round(demand_kg)),
                                    hour_of_day=DISPATCH_HOUR)
            p75 = round(p50 * 1.15, 1)   # rough P75 until we store per-aoi std
        else:
            p75 = 10.0
        dwell = p75

        # Failure probability from model or historical rate
        _, _, _, _, _, _, _, _, building_type, has_access, has_safe, \
            hour_rate, hist_rate, n_past, n_failed = row

        if fail_model:
            feats    = _make_stop_features(row, DISPATCH_HOUR + int(tw_open / 60), dwell)
            result   = fail_model.predict(feats)
            p_success = 1.0 - result["failure_probability"]
        else:
            p_success = hist_rate

        stops.append(Stop(
            stop_id          = f"S{i+1:02d}",
            address          = address,
            lat              = lat,
            lon              = lon,
            time_window_open = tw_open,
            time_window_close= tw_close,
            demand_kg        = demand_kg,
            dwell_mins       = dwell,
            p_success        = round(p_success, 3),
            aoi_type         = aoi_type,
            zone_id          = zone_id,
        ))

        risk = "HIGH" if p_success < 0.60 else ("MED" if p_success < 0.85 else "ok")
        print(f"  S{i+1:02d} {address:25s}  dwell={dwell:.0f}min  P(ok)={p_success:.0%}  [{risk}]")

    # ── Step 3: OSRM distance matrix ──────────────────────────────────────────
    from data.distance_matrix import _osrm_available
    osrm_ok = _osrm_available()
    backend  = "OSRM (real roads)" if osrm_ok else "Haversine (fallback — start Docker: docker start osrm-uk)"
    print(f"\n[3/4] Building {len(coords)}x{len(coords)} time matrix  [{backend}]...")
    try:
        matrix_list = build_matrix(coords)
        matrix = np.array(matrix_list, dtype=float)
        print(f"  Sample: depot->S01 = {matrix[0,1]:.1f} min, S01->S02 = {matrix[1,2]:.1f} min")
    except Exception as e:
        print(f"  Matrix build failed ({e})")
        return

    # ── Step 4: solve ─────────────────────────────────────────────────────────
    print(f"\n[4/4] Solving VRPTW with PyVRP ({NUM_VEHICLES} vehicles, {SHIFT_DURATION_MINS}min shift)...")
    result = solve_vrp(
        stops               = stops,
        time_matrix         = matrix,
        num_vehicles        = NUM_VEHICLES,
        shift_duration_mins = SHIFT_DURATION_MINS,
        apply_zone_penalty  = True,
        deferral_enabled    = True,
    )

    if result is None:
        print("  No feasible solution found.")
        return

    # ── Results ───────────────────────────────────────────────────────────────
    dispatch_dt = datetime.now().replace(hour=DISPATCH_HOUR, minute=0, second=0, microsecond=0)

    print("\n" + "=" * 65)
    print("SOLUTION")
    print("=" * 65)
    print(f"Vehicles used:   {result.num_vehicles} / {NUM_VEHICLES}")
    print(f"Total route time:{result.total_time:.0f} min")
    print(f"Expected FADR:   {result.expected_fadr:.1%}")
    print(f"Cascade risk:    {'YES — recheck buffer' if result.cascade_risk else 'No'}")
    print(f"Routing backend: {backend}")

    print("\nCost breakdown:")
    cb = result.cost_breakdown
    print(f"  Travel time (alpha): {cb['alpha_travel_time_mins']:.0f} min-equivalent")
    print(f"  Vehicles    (beta):  ${cb['beta_vehicles_usd']:.0f}")
    print(f"  Failure risk(gamma): EUR {cb['gamma_failures_eur']:.1f}")
    print(f"  Total weighted:      {cb['total_equivalent']:.0f}")

    if result.deferred:
        print(f"\nDeferred ({len(result.deferred)} stops — NOT loaded today):")
        for s in result.deferred:
            print(f"  {s.stop_id} {s.address:25s}  P(ok)={s.p_success:.0%}  "
                  f"window={s.time_window_open}-{s.time_window_close}min")
        print("  Action: retry tomorrow in a different time band")

    for route in result.routes:
        print(f"\n--- Vehicle {route.vehicle_id + 1}  "
              f"{route.total_time_mins:.0f}min  FADR={route.expected_fadr:.0%}  "
              f"{'CASCADE RISK' if route.cascade_risk else ''}")
        for step in route.steps:
            eta = dispatch_dt + timedelta(minutes=step.arrival_min)
            print(f"  {step.sequence}. {step.address:25s}"
                  f"  ETA {eta.strftime('%H:%M')}"
                  f"  dwell {step.dwell_mins:.0f}min"
                  f"  P(ok)={step.p_success:.0%}")

    print("\n" + "=" * 65)
    print("Pipeline complete.")
    print("=" * 65)


if __name__ == "__main__":
    run_pipeline()
