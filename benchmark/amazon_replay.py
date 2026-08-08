"""
benchmark/amazon_replay.py

Counterfactual replay on Amazon Last Mile Routing Research Challenge 2021.

Ground truth: actual driver sequences (6,112 US routes).
Ours: zone-coherent nearest-neighbour with OSRM road times.

Metric: total route travel time (minutes) — lower is better.
No time windows in the free dataset, so we optimise pure travel time (TSP proxy).

Usage:
    python -m benchmark.amazon_replay [--n 50] [--score High]
"""

import json
import logging
import random
import argparse
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

DATA_DIR   = Path("data/amazon")
ROUTE_FILE = DATA_DIR / "route_data.json"
SEQ_FILE   = DATA_DIR / "actual_sequences.json"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _route_travel_time(order: list, mat: np.ndarray) -> float:
    """Total travel time for a stop sequence (depot=0, stops=1..n)."""
    if not order:
        return 0.0
    t = mat[0, order[0]]
    for i in range(1, len(order)):
        t += mat[order[i-1], order[i]]
    t += mat[order[-1], 0]
    return float(t)


def _nearest_neighbour(n: int, mat: np.ndarray) -> list:
    """Greedy NN from depot."""
    unvisited = set(range(1, n + 1))
    order, cur = [], 0
    while unvisited:
        nxt = min(unvisited, key=lambda j: mat[cur, j])
        order.append(nxt)
        unvisited.remove(nxt)
        cur = nxt
    return order


def _zone_coherent_nn(stop_ids: list, zone_ids: list, mat: np.ndarray) -> list:
    """
    Zone-coherent nearest-neighbour: prefer staying in same zone.
    Zone penalty: add 2 min to cross-zone moves (same as VRP solver).
    """
    n = len(stop_ids)
    zone_pen = 2.0

    penalised = mat.copy().astype(float)
    for i in range(n + 1):
        zi = None if i == 0 else zone_ids[i - 1]
        for j in range(n + 1):
            if i == j:
                continue
            zj = None if j == 0 else zone_ids[j - 1]
            if zi is not None and zj is not None and zi != zj:
                penalised[i, j] += zone_pen

    return _nearest_neighbour(n, penalised)


def _build_matrix_osrm(coords: list) -> np.ndarray:
    from data.distance_matrix import build_matrix
    return np.array(build_matrix(coords), dtype=float)


def _pyvrp_order(stop_ids: list, zone_ids: list,
                 coords: list, mat: np.ndarray) -> list:
    """
    Run PyVRP as a TSP (single vehicle, loose windows, no deferral).
    Returns ordered list of matrix indices (1..n), same convention as other algos.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from optimizer.vrp_solver import Stop, solve_vrp

    n = len(stop_ids)
    stops = [
        Stop(
            stop_id           = sid,
            address           = zone_ids[i],
            lat               = coords[i + 1][0],   # coords[0]=depot, stops at 1..n
            lon               = coords[i + 1][1],
            time_window_open  = 0,
            time_window_close = 720,                 # unconstrained (12h)
            demand_kg         = 1.0,
            dwell_mins        = 1.0,                 # minimal — pure travel comparison
            p_success         = 1.0,                 # no deferral
            zone_id           = zone_ids[i],
        )
        for i, sid in enumerate(stop_ids)
    ]

    result = solve_vrp(
        stops               = stops,
        time_matrix         = mat,
        num_vehicles        = 1,
        vehicle_capacity_kg = 10000,
        shift_duration_mins = 720,
        apply_zone_penalty  = True,
        deferral_enabled    = False,
    )

    if result is None or not result.routes:
        return _nearest_neighbour(n, mat)   # fallback

    id_to_idx = {sid: i + 1 for i, sid in enumerate(stop_ids)}
    return [id_to_idx[step.stop_id] for step in result.routes[0].steps]


# ── Main replay ───────────────────────────────────────────────────────────────

def run_amazon_replay(n_routes: int = 50, score_filter: str = "High", seed: int = 42):
    print(f"\nAMAZON LAST MILE — counterfactual replay")
    print(f"Score filter: {score_filter} | Routes: {n_routes} | Matrix: OSRM\n")

    with open(ROUTE_FILE) as f:
        routes = json.load(f)
    with open(SEQ_FILE) as f:
        seqs = json.load(f)

    # Filter by route_score and minimum stop count
    candidates = [
        rid for rid, r in routes.items()
        if r.get("route_score") == score_filter
        and len([s for s in r["stops"].values() if s["type"] == "Dropoff"]) >= 10
        and rid in seqs
    ]
    print(f"Eligible routes ({score_filter} score, >=10 stops): {len(candidates)}")

    rng = random.Random(seed)
    sample = rng.sample(candidates, min(n_routes, len(candidates)))

    amazon_times, ours_times, nn_times, pyvrp_times = [], [], [], []
    skipped = 0

    print(f"{'Route':>6}  {'Stops':>5}  {'Amazon(min)':>11}  {'NaiveNN(min)':>12}  "
          f"{'Ours(min)':>10}  {'PyVRP(min)':>11}  {'PyVRP gain':>11}")
    print("-" * 90)

    for idx, rid in enumerate(sample):
        r      = routes[rid]
        stops  = r["stops"]

        # Depot = Station stop
        depot_stop = next(
            (s for s in stops.values() if s["type"] == "Station"),
            None
        )
        if depot_stop is None:
            skipped += 1
            continue

        # Delivery stops only
        delivery = {sid: s for sid, s in stops.items() if s["type"] == "Dropoff"}
        n = len(delivery)
        if n < 5:
            skipped += 1
            continue

        stop_ids  = list(delivery.keys())
        zone_ids  = [delivery[sid]["zone_id"] for sid in stop_ids]
        coords    = [(depot_stop["lat"], depot_stop["lng"])] + \
                    [(delivery[sid]["lat"], delivery[sid]["lng"]) for sid in stop_ids]

        try:
            mat = _build_matrix_osrm(coords)
        except Exception as e:
            skipped += 1
            continue

        np.fill_diagonal(mat, 0.0)

        # Amazon actual sequence: {stop_id: position}, position 0 = depot
        actual_seq = seqs[rid]["actual"]
        amazon_order = []
        for sid in stop_ids:
            if sid in actual_seq:
                amazon_order.append((actual_seq[sid], stop_ids.index(sid) + 1))
        amazon_order.sort()
        amazon_order = [x[1] for x in amazon_order]

        if not amazon_order:
            skipped += 1
            continue

        # Naive NN (no zone awareness)
        nn_order    = _nearest_neighbour(n, mat)

        # Ours: zone-coherent NN
        our_order   = _zone_coherent_nn(stop_ids, zone_ids, mat)

        # PyVRP: metaheuristic TSP with zone penalty
        pyvrp_order = _pyvrp_order(stop_ids, zone_ids, coords, mat)

        t_amazon = _route_travel_time(amazon_order, mat)
        t_nn     = _route_travel_time(nn_order,     mat)
        t_ours   = _route_travel_time(our_order,    mat)
        t_pyvrp  = _route_travel_time(pyvrp_order,  mat)

        amazon_times.append(t_amazon)
        nn_times.append(t_nn)
        ours_times.append(t_ours)
        pyvrp_times.append(t_pyvrp)

        gain_p = t_amazon - t_pyvrp
        pct_p  = gain_p / t_amazon * 100 if t_amazon > 0 else 0
        print(f"{idx:>6}  {n:>5}  {t_amazon:>11.1f}  {t_nn:>12.1f}  {t_ours:>10.1f}  "
              f"{t_pyvrp:>11.1f}  {gain_p:>+7.1f}min ({pct_p:+.1f}%)")

    if not amazon_times:
        print("No routes processed.")
        return

    print("\n" + "=" * 75)
    print("AGGREGATE")
    print("=" * 75)

    a  = np.mean(amazon_times)
    nn = np.mean(nn_times)
    o  = np.mean(ours_times)
    p  = np.mean(pyvrp_times)

    print(f"{'Metric':<35} {'Amazon':>10} {'Naive NN':>10} {'Ours(NN)':>10} {'PyVRP':>10}")
    print("-" * 80)
    print(f"{'Mean travel time (min)':<35} {a:>10.1f} {nn:>10.1f} {o:>10.1f} {p:>10.1f}")
    print(f"{'Median travel time (min)':<35} "
          f"{np.median(amazon_times):>10.1f} "
          f"{np.median(nn_times):>10.1f} "
          f"{np.median(ours_times):>10.1f} "
          f"{np.median(pyvrp_times):>10.1f}")
    print(f"{'Routes beating Amazon':<35} "
          f"{'':>10} {'':>10} "
          f"{sum(o < a for o, a in zip(ours_times, amazon_times)):>10} / {len(ours_times)}"
          f"  {sum(pv < a for pv, a in zip(pyvrp_times, amazon_times)):>4} / {len(pyvrp_times)}")

    gain_nn_vs_amazon    = (a - o) / a * 100
    gain_pyvrp_vs_amazon = (a - p) / a * 100
    gain_pyvrp_vs_nn     = (o - p) / o * 100

    print(f"\nHeadline:")
    print(f"  Zone-coherent NN vs Amazon: {gain_nn_vs_amazon:+.1f}% travel time")
    print(f"  PyVRP          vs Amazon:   {gain_pyvrp_vs_amazon:+.1f}% travel time")
    print(f"  PyVRP          vs Ours(NN): {gain_pyvrp_vs_nn:+.1f}% travel time")
    print(f"  Routes sampled:             {len(ours_times)} ({skipped} skipped)")
    print(f"  Score filter:               {score_filter}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",     type=int, default=50)
    parser.add_argument("--score", type=str, default="High",
                        choices=["High", "Medium", "Low"])
    parser.add_argument("--seed",  type=int, default=42)
    args = parser.parse_args()
    run_amazon_replay(n_routes=args.n, score_filter=args.score, seed=args.seed)
