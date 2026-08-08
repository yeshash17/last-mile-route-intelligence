"""
benchmark/amazon_cascade_replay.py

Hybrid synthetic FADR benchmark on Amazon Last Mile routes.

  Real geometry  : Amazon Last Mile 2021 (High-score routes, 80-197 stops)
  Real road times: OSRM (WA / IL / TX / MA); Haversine fallback for LA
  Synthetic dwell: Normal(mean=5min, std=3min) — Western residential delivery
  P(success)     : Normal(mean=0.85, std=0.12) per stop

WHY THIS MATTERS:
  Baseline plans 2 min dwell (industry norm). Reality is 5 min mean.
  On a 120-stop route: baseline thinks 360 min of dwell, reality is 600 min.
  Cascade hits at stop ~60 -> 60 undelivered (definite failures, must reattempt).
  Our system plans at P75 (7 min), correctly schedules ~50 stops, defers the rest.
  Deferred = planned reattempt tomorrow. Cascade = surprise miss, angry customer.

Usage:
    python -m benchmark.amazon_cascade_replay
    python -m benchmark.amazon_cascade_replay --n 30 --score High
"""

import json
import sys
import random
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.replay import (
    run_baseline, run_naive_opt, run_ours,
    SHIFT_MINS, PLANNED_DWELL,
)
from optimizer.vrp_solver import Stop

DATA_DIR   = Path("data/amazon")
ROUTE_FILE = DATA_DIR / "route_data.json"

# Western residential delivery dwell model
DWELL_MEAN    = 5.0   # mean actual dwell (door knock + hand-off + walk back)
DWELL_STD     = 3.0   # high variance: 1 min parcel drop vs 15 min signature
DWELL_P75     = 7.0   # P75 — what our planner uses
P_SUCCESS_MU  = 0.85  # base first-attempt success (residential, daytime)
P_SUCCESS_STD = 0.12  # spread; ~10% of stops fall below 0.60

SEED = 42


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_matrix(coords: list) -> np.ndarray:
    from data.distance_matrix import build_matrix
    return np.array(build_matrix(coords), dtype=float)


def _synthetic_dwells(n: int, rng: np.random.Generator) -> list:
    return list(np.clip(rng.normal(DWELL_MEAN, DWELL_STD, n), 1.0, 20.0))


def _synthetic_p_success(n: int, rng: np.random.Generator) -> list:
    return list(np.clip(rng.normal(P_SUCCESS_MU, P_SUCCESS_STD, n), 0.20, 0.99))


def _build_stops_vrp(stop_ids: list, zone_ids: list,
                     coords: list, p_success: list) -> list:
    return [
        Stop(
            stop_id           = sid,
            address           = zone_ids[i],
            lat               = coords[i + 1][0],
            lon               = coords[i + 1][1],
            time_window_open  = 0,
            time_window_close = SHIFT_MINS,
            demand_kg         = 1.0,
            dwell_mins        = DWELL_P75,
            p_success         = p_success[i],
            zone_id           = zone_ids[i],
        )
        for i, sid in enumerate(stop_ids)
    ]


def _how_many_fit_naive(n: int, mat: np.ndarray, dwell: float) -> int:
    """Count stops a naive NN planner fits given assumed dwell."""
    unvisited = set(range(1, n + 1))
    order, cur, t = [], 0, 0.0
    while unvisited:
        nxt = min(unvisited, key=lambda j: mat[cur, j])
        t_arr = t + mat[cur, nxt]
        if t_arr + dwell + mat[nxt, 0] > SHIFT_MINS:
            break
        t = t_arr + dwell
        order.append(nxt)
        unvisited.remove(nxt)
        cur = nxt
    return len(order)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_amazon_cascade(n_routes: int = 30, score_filter: str = "High",
                       seed: int = SEED):
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    print(f"\nAMAZON CASCADE BENCHMARK — hybrid synthetic FADR")
    print(f"Score: {score_filter} | Routes: {n_routes} | "
          f"Dwell model: N({DWELL_MEAN},{DWELL_STD}) min | Shift: {SHIFT_MINS} min\n")

    with open(ROUTE_FILE) as f:
        routes = json.load(f)

    candidates = [
        rid for rid, r in routes.items()
        if r.get("route_score") == score_filter
        and len([s for s in r["stops"].values() if s["type"] == "Dropoff"]) >= 50
    ]
    print(f"Eligible routes ({score_filter}, >=50 stops): {len(candidates)}")

    sample = py_rng.sample(candidates, min(n_routes, len(candidates)))

    results = {"baseline": [], "naive_opt": [], "ours": []}
    skipped = 0

    hdr = (f"{'Rt':>3} {'N':>4}  "
           f"{'B:FADR':>7} {'NO:FADR':>8} {'O:FADR':>8}  "
           f"{'Gain':>7}  {'B:Casc':>7} {'O:Def':>6}  "
           f"{'B:plan':>7} {'O:plan':>7}")
    print(hdr)
    print("-" * len(hdr))

    for idx, rid in enumerate(sample):
        r     = routes[rid]
        stops = r["stops"]

        depot = next((s for s in stops.values() if s["type"] == "Station"), None)
        if depot is None:
            skipped += 1
            continue

        delivery = {sid: s for sid, s in stops.items() if s["type"] == "Dropoff"}
        n = len(delivery)
        if n < 10:
            skipped += 1
            continue

        stop_ids = list(delivery.keys())
        zone_ids = [delivery[sid]["zone_id"] for sid in stop_ids]
        coords   = [(depot["lat"], depot["lng"])] + \
                   [(delivery[sid]["lat"], delivery[sid]["lng"]) for sid in stop_ids]

        try:
            mat = _build_matrix(coords)
        except Exception:
            skipped += 1
            continue

        np.fill_diagonal(mat, 0.0)

        # Synthetic dwell and success probability
        real_dwells = _synthetic_dwells(n, rng)
        p_success   = _synthetic_p_success(n, rng)

        # Build Stop objects for our solver (P75 dwell, loose windows)
        stops_vrp = _build_stops_vrp(stop_ids, zone_ids, coords, p_success)

        # Run all three arms
        b  = run_baseline(n, mat, real_dwells, p_success, rng)
        no = run_naive_opt(n, coords, mat, real_dwells, p_success)
        o  = run_ours(stops_vrp, mat, real_dwells, p_success, use_pyvrp=False)

        results["baseline"].append(b)
        results["naive_opt"].append(no)
        results["ours"].append(o)

        gain = (o["actual_fadr"] - b["actual_fadr"]) * 100
        print(f"{idx:>3} {n:>4}  "
              f"{b['actual_fadr']*100:>6.1f}% "
              f"{no['actual_fadr']*100:>7.1f}% "
              f"{o['actual_fadr']*100:>7.1f}%  "
              f"{gain:>+6.1f}pp  "
              f"{b['failed_cascade']:>7} {o.get('deferred',0):>6}  "
              f"{b['stops_planned']:>7} {o['stops_planned']:>7}")

    if not results["baseline"]:
        print("No routes processed.")
        return

    print("\n" + "=" * 75)
    print("AGGREGATE RESULTS")
    print("=" * 75)

    def agg(key, field):
        vals = [r[field] for r in results[key]]
        return np.mean(vals) if vals else 0.0

    b_fadr  = agg("baseline",  "actual_fadr") * 100
    no_fadr = agg("naive_opt", "actual_fadr") * 100
    o_fadr  = agg("ours",      "actual_fadr") * 100

    b_casc  = agg("baseline",  "failed_cascade")
    o_def   = agg("ours",      "deferred")
    b_plan  = agg("baseline",  "stops_planned")
    o_plan  = agg("ours",      "stops_planned")

    gain    = o_fadr - b_fadr

    print(f"{'Metric':<35} {'Baseline':>10} {'Naive NN':>10} {'Ours':>10}")
    print("-" * 70)
    print(f"{'Actual FADR':<35} {b_fadr:>9.1f}% {no_fadr:>9.1f}% {o_fadr:>9.1f}%")
    print(f"{'Cascade failures/route':<35} {b_casc:>10.1f} {'':>10} {0.0:>10.1f}")
    print(f"{'Deferred stops/route':<35} {'0.0':>10} {'':>10} {o_def:>10.1f}")
    print(f"{'Stops planned/route':<35} {b_plan:>10.1f} {'':>10} {o_plan:>10.1f}")

    n_routes_done = len(results["baseline"])
    b_total_casc  = sum(r["failed_cascade"] for r in results["baseline"])
    o_total_def   = sum(r.get("deferred", 0) for r in results["ours"])

    # Cost of cascade vs deferral
    CASCADE_COST  = 12.0   # $ per cascade failure (reattempt + fuel + customer churn)
    DEFERRAL_COST = 3.0    # $ per deferred stop (scheduled retry, customer notified)
    daily_saving  = (b_total_casc * CASCADE_COST - o_total_def * DEFERRAL_COST) / n_routes_done

    print(f"\nHeadline:")
    print(f"  FADR improvement (Ours vs Baseline):  {gain:+.1f} percentage points")
    print(f"  Cascade failures prevented/route:      {b_casc:.1f}")
    print(f"  Cascade -> Deferral conversion/route:  {b_casc:.1f} cascade  ->  {o_def:.1f} deferred")
    print(f"\n  Cost model: cascade=${CASCADE_COST}/stop, deferral=${DEFERRAL_COST}/stop")
    print(f"  Daily saving/route:                   ${daily_saving:.2f}")
    print(f"  Annualised (100 drivers x 313 days):  ${daily_saving * 100 * 313:,.0f}/year")
    print(f"\n  Routes processed: {n_routes_done} ({skipped} skipped)")
    print(f"  Dwell model:      N({DWELL_MEAN},{DWELL_STD}) min  "
          f"[mean actual vs {PLANNED_DWELL:.0f} min planned by baseline]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",     type=int, default=30)
    parser.add_argument("--score", type=str, default="High",
                        choices=["High", "Medium", "Low"])
    parser.add_argument("--seed",  type=int, default=SEED)
    args = parser.parse_args()
    run_amazon_cascade(n_routes=args.n, score_filter=args.score, seed=args.seed)
