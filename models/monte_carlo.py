"""
models/monte_carlo.py

Monte Carlo FADR simulation — the SIMULATE step of the DI cycle.

Given a planned route and a dwell predictor, run N scenarios by sampling
dwell times from each stop's predicted distribution.  Outputs:

    mean_fadr   — expected first-attempt delivery rate
    p10_fadr    — worst-case FADR (10th percentile across scenarios)
    p90_fadr    — best-case  FADR (90th percentile)
    p_cascade   — P(any cascade failure occurs) across scenarios
    cascade_at  — expected stop index where first cascade occurs (if any)
    risk_label  — "LOW" / "MEDIUM" / "HIGH" based on p_cascade

Usage:
    from models.monte_carlo import simulate_route_risk
    from models.service_time import ServiceTimeModel

    model = ServiceTimeModel.load()
    stops = [...]        # list of dicts with aoi_id, aoi_type, package_count, hour_of_day
    result = simulate_route_risk(stops, time_matrix, model, n_sim=500)
    print(result)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

DEFAULT_SHIFT_MINS = 480.0   # 8-hour driver shift


@dataclass
class RouteRisk:
    mean_fadr:    float   # E[FADR] across simulations
    p10_fadr:     float   # 10th-pct FADR (bad day)
    p90_fadr:     float   # 90th-pct FADR (good day)
    p_cascade:      float          # P(≥1 cascade failure)
    cascade_at:     Optional[float]  # mean stop index where cascade starts
    mean_cascade_n: float          # mean number of stops cascaded per scenario
    risk_label:     str            # LOW / MEDIUM / HIGH
    n_sim:          int
    n_stops:        int
    mean_dwell:     float          # mean predicted dwell across stops


def _simulate_one(
    order:       np.ndarray,   # stop indices 0..n-1
    mat:         np.ndarray,   # (n+1) x (n+1) travel time matrix, depot=0
    dwells:      np.ndarray,   # sampled dwell per stop (n,)
    delivered:   np.ndarray,   # Bernoulli sample: 1=delivered, 0=failed (n,)
    shift_mins:  float,
) -> tuple[float, int]:
    """
    Drive the route once with given dwells + sampled delivery outcomes.
    Return (actual_fadr, cascade_start_idx). cascade_start_idx = -1 if no cascade.
    """
    t = 0.0
    n_completed  = 0
    n_delivered  = 0
    cascade_start = -1

    for pos, i in enumerate(order):
        prev  = 0 if pos == 0 else order[pos - 1] + 1
        t_arr = t + mat[prev, i + 1]
        if t_arr >= shift_mins:
            cascade_start = pos
            break
        t = t_arr + dwells[i]
        n_completed += 1
        n_delivered += int(delivered[i])

    total     = len(order)
    actual_fadr = n_delivered / total if total > 0 else 0.0
    return actual_fadr, cascade_start


def simulate_route_risk(
    stops:       list,          # list of dicts: aoi_id, aoi_type, package_count, hour_of_day, p_success
    time_matrix: np.ndarray,    # (n+1)x(n+1) with depot at row/col 0
    dwell_model,                # ServiceTimeModel instance
    order:       Optional[list] = None,   # stop visit order (indices 0..n-1); None = sequential
    n_sim:       int = 500,
    seed:        int = 42,
    shift_mins:  float = DEFAULT_SHIFT_MINS,
) -> RouteRisk:
    """
    Monte Carlo FADR simulation.

    Parameters
    ----------
    stops       : list of stop dicts (aoi_id, aoi_type, package_count, hour_of_day, p_success)
    time_matrix : OSRM or Haversine (n+1)x(n+1), depot at index 0
    dwell_model : ServiceTimeModel with predict_with_std()
    order       : visit sequence (indices into stops); None = 0,1,...,n-1
    n_sim       : Monte Carlo iterations
    seed        : RNG seed

    Returns
    -------
    RouteRisk dataclass with FADR distribution and risk label
    """
    rng   = np.random.default_rng(seed)
    n     = len(stops)
    order = np.array(order if order is not None else list(range(n)))

    # Pre-compute dwell mean + std per stop from the model
    means = np.zeros(n)
    stds  = np.zeros(n)
    for i, s in enumerate(stops):
        mu, sigma = dwell_model.predict_with_std(
            aoi_id        = int(s.get("aoi_id", -1)),
            aoi_type      = int(s.get("aoi_type", 1)),
            package_count = int(s.get("package_count", 1)),
            hour_of_day   = int(s.get("hour_of_day", 10)),
        )
        means[i] = mu
        stds[i]  = sigma

    p_success = np.array([float(s.get("p_success", 0.85)) for s in stops])

    # Run simulations
    fadrs           = np.zeros(n_sim)
    cascade_starts  = []
    cascade_counts  = []   # how many stops cascaded per scenario

    ratio  = np.maximum(stds / np.maximum(means, 0.1), 0.01)
    sig_ln = np.sqrt(np.log(1 + ratio ** 2))
    mu_ln  = np.log(np.maximum(means, 0.1)) - 0.5 * sig_ln ** 2

    for sim in range(n_sim):
        dwells    = np.clip(rng.lognormal(mu_ln, sig_ln), 0.5, 60.0)
        delivered = (rng.random(n) < p_success).astype(int)   # Bernoulli per stop

        fadr, cs  = _simulate_one(order, time_matrix, dwells, delivered, shift_mins)
        fadrs[sim] = fadr
        if cs >= 0:
            cascade_starts.append(cs)
            cascade_counts.append(len(order) - cs)
        else:
            cascade_counts.append(0)

    p_cascade      = len(cascade_starts) / n_sim
    cascade_at     = float(np.mean(cascade_starts)) if cascade_starts else None
    mean_cascade_n = float(np.mean(cascade_counts))  # avg stops cascaded across all scenarios

    if p_cascade < 0.10:
        label = "LOW"
    elif p_cascade < 0.30:
        label = "MEDIUM"
    else:
        label = "HIGH"

    return RouteRisk(
        mean_fadr      = round(float(np.mean(fadrs)), 4),
        p10_fadr       = round(float(np.percentile(fadrs, 10)), 4),
        p90_fadr       = round(float(np.percentile(fadrs, 90)), 4),
        p_cascade      = round(p_cascade, 4),
        cascade_at     = round(cascade_at, 1) if cascade_at is not None else None,
        mean_cascade_n = round(mean_cascade_n, 1),
        risk_label     = label,
        n_sim          = n_sim,
        n_stops        = n,
        mean_dwell     = round(float(means.mean()), 2),
    )


def compare_plans(
    stops:       list,
    time_matrix: np.ndarray,
    dwell_model,
    plans:       dict,          # {"plan_name": [order_indices], ...}
    n_sim:       int = 500,
) -> dict:
    """
    Compare multiple route orderings under uncertainty.

    Returns dict of plan_name → RouteRisk, sorted by mean_fadr descending.
    """
    results = {}
    for name, order in plans.items():
        results[name] = simulate_route_risk(
            stops, time_matrix, dwell_model, order=order, n_sim=n_sim
        )
    return dict(sorted(results.items(), key=lambda x: x[1].mean_fadr, reverse=True))


def print_risk_report(plan_name: str, risk: RouteRisk):
    bar_p    = "#" * int(risk.p_cascade * 20)
    bar_fadr = "#" * int(risk.mean_fadr * 20)
    casc_str = (f"stop ~{risk.cascade_at:.0f}" if risk.cascade_at else "none")
    print(f"\n{'─'*55}")
    print(f"  Plan: {plan_name}  [{risk.risk_label} RISK]  n={risk.n_stops} stops")
    print(f"{'─'*55}")
    print(f"  Mean FADR:     {risk.mean_fadr*100:5.1f}%  {bar_fadr}")
    print(f"  FADR P10/P90:  {risk.p10_fadr*100:.1f}% – {risk.p90_fadr*100:.1f}%")
    print(f"  P(cascade):    {risk.p_cascade*100:5.1f}%  {bar_p}")
    print(f"  Cascade at:    {casc_str}")
    print(f"  Mean dwell:    {risk.mean_dwell:.1f} min/stop")


# ── CLI demo ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import json
    import argparse
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from models.service_time import ServiceTimeModel
    from data.distance_matrix import build_matrix

    parser = argparse.ArgumentParser(description="Monte Carlo route risk demo")
    parser.add_argument("--city",   default="sh")
    parser.add_argument("--n",      type=int, default=25, help="stops per route")
    parser.add_argument("--nsim",   type=int, default=500)
    args = parser.parse_args()

    print(f"Loading service time model...")
    model = ServiceTimeModel.load()
    print(f"  Global mean dwell: {model.global_mean:.1f} min")
    print(f"  Model MAE (L3):    {model.level_mae.get('L3_per_aoi', '?')} min\n")

    # Build a synthetic demo route
    from data.loader import load_lade
    print(f"Loading LaDe-{args.city.upper()} for demo route...")
    df = load_lade(city=args.city)

    # Sample one courier-day with enough stops
    groups = df.groupby(["courier_id", "ds"])
    candidates = [(k, v) for k, v in groups if len(v) >= args.n]
    if not candidates:
        print("No routes found. Try --n smaller.")
        sys.exit(1)

    rng = np.random.default_rng(42)
    key, route_df = candidates[rng.integers(len(candidates))]
    route_df = route_df.head(args.n).reset_index(drop=True)

    print(f"Route: courier={key[0]}, date={key[1]}, stops={len(route_df)}")

    # Coords: first row is treated as depot (station or first stop)
    coords = [(row["lat"], row["lng"]) for _, row in route_df.iterrows()]
    depot  = coords[0]
    coords = [depot] + coords[1:]

    print("Building travel time matrix (OSRM / Haversine)...")
    mat = np.array(build_matrix(coords), dtype=float)
    np.fill_diagonal(mat, 0.0)

    stops = [
        {
            "aoi_id":        int(row["aoi_id"]),
            "aoi_type":      int(row["aoi_type"]),
            "package_count": int(row["package_count"]),
            "hour_of_day":   row["delivery_dt"].hour,
            "p_success":     0.85,
        }
        for _, row in route_df.iloc[1:].iterrows()
    ]
    n_stops = len(stops)

    # Three plans to compare
    seq_order = list(range(n_stops))
    nn_order  = list(rng.permutation(n_stops))   # random as stand-in for NN

    plans = {
        "Sequential (baseline)": seq_order,
        "Random NN":             nn_order,
    }

    print(f"\nRunning {args.nsim} Monte Carlo simulations per plan...\n")
    results = compare_plans(stops, mat, model, plans, n_sim=args.nsim)

    for name, risk in results.items():
        print_risk_report(name, risk)

    print(f"\n{'═'*55}")
    best = next(iter(results))
    best_r = results[best]
    print(f"RECOMMENDATION: '{best}'")
    print(f"  Expected FADR {best_r.mean_fadr*100:.1f}%  "
          f"[{best_r.p10_fadr*100:.0f}–{best_r.p90_fadr*100:.0f}%]  "
          f"cascade risk {best_r.p_cascade*100:.0f}%")
