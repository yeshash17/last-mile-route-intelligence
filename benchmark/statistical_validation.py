"""
benchmark/statistical_validation.py

Statistical rigor layer for benchmark results.

  Part 1 — Bootstrap 95% CI on cascade FADR gain (30 Amazon routes)
  Part 2 — Bootstrap 95% CI on PyVRP travel time gain (50 Amazon routes)
  Part 3 — Sensitivity sweep: FADR gain vs baseline dwell assumption (1–8 min)

Converts point estimates into defensible statistical claims:
  "15.7pp [95% CI: 12.1–19.3pp, p<0.001, n=30]"

Usage:
    python -m benchmark.statistical_validation
    python -m benchmark.statistical_validation --skip-travel  # cascade + sensitivity only
"""

import json
import sys
import random
import argparse
import numpy as np
from pathlib import Path
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.replay import (
    run_baseline, run_naive_opt, run_ours,
    _nearest_neighbour_order, _simulate_execution,
    SHIFT_MINS,
)
from benchmark.amazon_cascade_replay import (
    _build_matrix, _synthetic_dwells, _synthetic_p_success,
    _build_stops_vrp, DWELL_MEAN, DWELL_STD, DWELL_P75,
)
from benchmark.amazon_replay import (
    _build_matrix_osrm, _nearest_neighbour, _zone_coherent_nn,
    _route_travel_time,
)

DATA_DIR   = Path("data/amazon")
ROUTE_FILE = DATA_DIR / "route_data.json"
SEQ_FILE   = DATA_DIR / "actual_sequences.json"

SEED       = 42
N_BOOT     = 1000


# ── Bootstrap CI ─────────────────────────────────────────────────────────────

def bootstrap_ci(values: list, n_boot: int = N_BOOT,
                 ci: float = 0.95) -> tuple[float, float, float]:
    """
    Bootstrap confidence interval for mean of values.
    Returns (mean, lo, hi).
    """
    arr  = np.array(values)
    mean = float(arr.mean())
    boots = [
        np.mean(np.random.choice(arr, size=len(arr), replace=True))
        for _ in range(n_boot)
    ]
    alpha = (1 - ci) / 2
    lo = float(np.percentile(boots, alpha * 100))
    hi = float(np.percentile(boots, (1 - alpha) * 100))
    return mean, lo, hi


def wilcoxon_p(a: list, b: list) -> float:
    """Paired Wilcoxon signed-rank test. Returns p-value."""
    try:
        _, p = scipy_stats.wilcoxon(a, b, alternative="greater")
        return float(p)
    except Exception:
        return float("nan")


# ── Part 1: Cascade FADR bootstrap ───────────────────────────────────────────

def run_cascade_bootstrap(n_routes: int = 30, score: str = "High") -> dict:
    print(f"\n{'='*62}")
    print(f"  PART 1 — Bootstrap CI: Cascade FADR Gain")
    print(f"  n_routes={n_routes}  score={score}  n_boot={N_BOOT}")
    print(f"{'='*62}")

    rng    = np.random.default_rng(SEED)
    py_rng = random.Random(SEED)

    with open(ROUTE_FILE) as f:
        routes = json.load(f)

    candidates = [
        rid for rid, r in routes.items()
        if r.get("route_score") == score
        and len([s for s in r["stops"].values() if s["type"] == "Dropoff"]) >= 50
    ]
    sample = py_rng.sample(candidates, min(n_routes, len(candidates)))

    b_fadrs, o_fadrs, gains = [], [], []

    for rid in sample:
        r     = routes[rid]
        stops = r["stops"]
        depot = next((s for s in stops.values() if s["type"] == "Station"), None)
        delivery = {sid: s for sid, s in stops.items() if s["type"] == "Dropoff"}
        if depot is None or len(delivery) < 10:
            continue

        stop_ids = list(delivery.keys())
        zone_ids = [delivery[sid]["zone_id"] for sid in stop_ids]
        n        = len(stop_ids)
        coords   = [(depot["lat"], depot["lng"])] + \
                   [(delivery[sid]["lat"], delivery[sid]["lng"]) for sid in stop_ids]

        try:
            mat = _build_matrix(coords)
        except Exception:
            continue

        np.fill_diagonal(mat, 0.0)
        real_dwells = _synthetic_dwells(n, rng)
        p_success   = _synthetic_p_success(n, rng)
        stops_vrp   = _build_stops_vrp(stop_ids, zone_ids, coords, p_success)

        b = run_baseline(n, mat, real_dwells, p_success, rng)
        o = run_ours(stops_vrp, mat, real_dwells, p_success, use_pyvrp=False)

        b_fadrs.append(b["actual_fadr"] * 100)
        o_fadrs.append(o["actual_fadr"] * 100)
        gains.append((o["actual_fadr"] - b["actual_fadr"]) * 100)

    n_valid = len(gains)
    mean_b, lo_b, hi_b = bootstrap_ci(b_fadrs)
    mean_o, lo_o, hi_o = bootstrap_ci(o_fadrs)
    mean_g, lo_g, hi_g = bootstrap_ci(gains)
    p_val = wilcoxon_p(o_fadrs, b_fadrs)

    print(f"\n  Routes processed: {n_valid}")
    print(f"\n  {'Metric':<30} {'Mean':>8} {'95% CI':>20}")
    print(f"  {'─'*30} {'─'*8} {'─'*20}")
    print(f"  {'Baseline FADR':<30} {mean_b:>7.1f}% [{lo_b:.1f}–{hi_b:.1f}%]")
    print(f"  {'Ours FADR':<30} {mean_o:>7.1f}% [{lo_o:.1f}–{hi_o:.1f}%]")
    print(f"  {'FADR Gain (Ours - Baseline)':<30} {mean_g:>+7.1f}pp [{lo_g:.1f}–{hi_g:.1f}pp]")
    print(f"\n  Wilcoxon p-value (one-sided): {p_val:.2e}")
    print(f"  Statistical significance: {'YES (p<0.05)' if p_val < 0.05 else 'NO'}")
    print(f"\n  CLAIM: +{mean_g:.1f}pp FADR [95% CI: {lo_g:.1f}–{hi_g:.1f}pp, "
          f"p={p_val:.1e}, n={n_valid}]")

    return {"gains": gains, "mean": mean_g, "lo": lo_g, "hi": hi_g, "p": p_val}


# ── Part 2: PyVRP travel time bootstrap ──────────────────────────────────────

def run_travel_bootstrap(n_routes: int = 50, score: str = "High") -> dict:
    print(f"\n{'='*62}")
    print(f"  PART 2 — Bootstrap CI: PyVRP Travel Time Gain vs Amazon")
    print(f"  n_routes={n_routes}  score={score}  n_boot={N_BOOT}")
    print(f"{'='*62}")

    from benchmark.amazon_replay import _pyvrp_order

    py_rng = random.Random(SEED)
    with open(ROUTE_FILE) as f:
        routes = json.load(f)
    with open(SEQ_FILE) as f:
        seqs = json.load(f)

    candidates = [
        rid for rid, r in routes.items()
        if r.get("route_score") == score
        and len([s for s in r["stops"].values() if s["type"] == "Dropoff"]) >= 10
        and rid in seqs
    ]
    sample = py_rng.sample(candidates, min(n_routes, len(candidates)))

    amazon_times, ours_times, pyvrp_times = [], [], []

    for idx, rid in enumerate(sample):
        r      = routes[rid]
        stops  = r["stops"]
        depot  = next((s for s in stops.values() if s["type"] == "Station"), None)
        if depot is None:
            continue
        delivery = {sid: s for sid, s in stops.items() if s["type"] == "Dropoff"}
        n = len(delivery)
        if n < 5:
            continue

        stop_ids  = list(delivery.keys())
        zone_ids  = [delivery[sid]["zone_id"] for sid in stop_ids]
        coords    = [(depot["lat"], depot["lng"])] + \
                    [(delivery[sid]["lat"], delivery[sid]["lng"]) for sid in stop_ids]
        try:
            mat = _build_matrix_osrm(coords)
        except Exception:
            continue
        np.fill_diagonal(mat, 0.0)

        actual_seq = seqs[rid]["actual"]
        amazon_order = []
        for sid in stop_ids:
            if sid in actual_seq:
                amazon_order.append((actual_seq[sid], stop_ids.index(sid) + 1))
        amazon_order.sort()
        amazon_order = [x[1] for x in amazon_order]
        if not amazon_order:
            continue

        our_order   = _zone_coherent_nn(stop_ids, zone_ids, mat)
        pyvrp_ord   = _pyvrp_order(stop_ids, zone_ids, coords, mat)

        t_amazon = _route_travel_time(amazon_order, mat)
        t_ours   = _route_travel_time(our_order, mat)
        t_pyvrp  = _route_travel_time(pyvrp_ord, mat)

        amazon_times.append(t_amazon)
        ours_times.append(t_ours)
        pyvrp_times.append(t_pyvrp)

        if (idx + 1) % 10 == 0:
            print(f"  ... {idx+1}/{len(sample)} routes done")

    nn_gains    = [(a - o) / a * 100 for a, o in zip(amazon_times, ours_times)]
    pyvrp_gains = [(a - p) / a * 100 for a, p in zip(amazon_times, pyvrp_times)]

    n_valid = len(nn_gains)
    m_nn, lo_nn, hi_nn       = bootstrap_ci(nn_gains)
    m_pv, lo_pv, hi_pv       = bootstrap_ci(pyvrp_gains)
    p_nn   = wilcoxon_p(ours_times,  amazon_times)   # ours < amazon (lower=better)
    p_pv   = wilcoxon_p(pyvrp_times, amazon_times)

    # Note: for travel time, "ours < amazon" means ours is better
    # wilcoxon_p uses "greater" so we flip: p_nn = P(amazon > ours)
    # Actually flip: compute P(ours < amazon) directly
    try:
        _, p_nn  = scipy_stats.wilcoxon(
            [a - o for a, o in zip(amazon_times, ours_times)],
            alternative="greater"
        )
        _, p_pv  = scipy_stats.wilcoxon(
            [a - p for a, p in zip(amazon_times, pyvrp_times)],
            alternative="greater"
        )
    except Exception:
        p_nn, p_pv = float("nan"), float("nan")

    print(f"\n  Routes processed: {n_valid}")
    print(f"\n  {'Metric':<38} {'Mean':>8} {'95% CI':>18}")
    print(f"  {'─'*38} {'─'*8} {'─'*18}")
    print(f"  {'Zone-coherent NN vs Amazon':<38} {m_nn:>+7.2f}% [{lo_nn:.2f}–{hi_nn:.2f}%]")
    print(f"  {'PyVRP vs Amazon':<38} {m_pv:>+7.2f}% [{lo_pv:.2f}–{hi_pv:.2f}%]")
    print(f"\n  Wilcoxon p (NN vs Amazon):    {p_nn:.2e}")
    print(f"  Wilcoxon p (PyVRP vs Amazon): {p_pv:.2e}")
    print(f"\n  CLAIM (NN):    {m_nn:+.1f}% travel time [95% CI: {lo_nn:.1f}–{hi_nn:.1f}%, "
          f"p={p_nn:.1e}, n={n_valid}]")
    print(f"  CLAIM (PyVRP): {m_pv:+.1f}% travel time [95% CI: {lo_pv:.1f}–{hi_pv:.1f}%, "
          f"p={p_pv:.1e}, n={n_valid}]")

    return {
        "nn":    {"mean": m_nn,  "lo": lo_nn, "hi": hi_nn, "p": p_nn},
        "pyvrp": {"mean": m_pv, "lo": lo_pv, "hi": hi_pv, "p": p_pv},
    }


# ── Part 3: Sensitivity sweep ─────────────────────────────────────────────────

def run_sensitivity_sweep(n_routes: int = 20, score: str = "High") -> dict:
    print(f"\n{'='*62}")
    print(f"  PART 3 — Sensitivity: FADR Gain vs Baseline Dwell Assumption")
    print(f"  Sweeping planned_dwell from 1.0 to 8.0 min (step 0.5)")
    print(f"  Fixed: actual dwell ~ N({DWELL_MEAN},{DWELL_STD})  ours plans at P75={DWELL_P75}min")
    print(f"{'='*62}")

    rng    = np.random.default_rng(SEED)
    py_rng = random.Random(SEED)

    with open(ROUTE_FILE) as f:
        routes = json.load(f)

    candidates = [
        rid for rid, r in routes.items()
        if r.get("route_score") == score
        and len([s for s in r["stops"].values() if s["type"] == "Dropoff"]) >= 50
    ]
    sample = py_rng.sample(candidates, min(n_routes, len(candidates)))

    # Pre-compute matrices + dwells once (expensive)
    print(f"\n  Pre-computing {len(sample)} route matrices...")
    route_data = []
    for rid in sample:
        r     = routes[rid]
        stops = r["stops"]
        depot = next((s for s in stops.values() if s["type"] == "Station"), None)
        delivery = {sid: s for sid, s in stops.items() if s["type"] == "Dropoff"}
        if depot is None or len(delivery) < 10:
            continue

        stop_ids = list(delivery.keys())
        zone_ids = [delivery[sid]["zone_id"] for sid in stop_ids]
        n        = len(stop_ids)
        coords   = [(depot["lat"], depot["lng"])] + \
                   [(delivery[sid]["lat"], delivery[sid]["lng"]) for sid in stop_ids]
        try:
            mat = _build_matrix(coords)
        except Exception:
            continue
        np.fill_diagonal(mat, 0.0)

        real_dwells = _synthetic_dwells(n, rng)
        p_success   = _synthetic_p_success(n, rng)
        stops_vrp   = _build_stops_vrp(stop_ids, zone_ids, coords, p_success)

        route_data.append({
            "n": n, "mat": mat, "dwells": real_dwells,
            "p_success": p_success, "stops_vrp": stops_vrp,
        })

    print(f"  Valid routes: {len(route_data)}")

    # Ours FADR is fixed (doesn't depend on baseline assumption)
    print(f"  Computing Ours FADR (fixed)...")
    ours_fadrs = []
    for rd in route_data:
        o = run_ours(rd["stops_vrp"], rd["mat"], rd["dwells"],
                     rd["p_success"], use_pyvrp=False)
        ours_fadrs.append(o["actual_fadr"] * 100)
    ours_mean = float(np.mean(ours_fadrs))

    # Sweep baseline dwell assumption
    sweep_values = np.arange(1.0, 8.5, 0.5)
    results = {}

    print(f"\n  {'Baseline dwell':>14}  {'Baseline FADR':>13}  {'Gain vs Ours':>13}  {'Crossover?':>10}")
    print(f"  {'─'*14}  {'─'*13}  {'─'*13}  {'─'*10}")

    for dwell_assumption in sweep_values:
        b_fadrs_sweep = []
        rng_sweep = np.random.default_rng(SEED + 1)   # fresh rng for reproducibility
        for rd in route_data:
            b = run_baseline(rd["n"], rd["mat"], rd["dwells"],
                             rd["p_success"], rng_sweep,
                             planned_dwell=dwell_assumption)
            b_fadrs_sweep.append(b["actual_fadr"] * 100)

        b_mean = float(np.mean(b_fadrs_sweep))
        gain   = ours_mean - b_mean
        crossed = gain <= 0.0
        marker = " <-- CROSSOVER" if crossed else ""

        print(f"  {dwell_assumption:>14.1f}  {b_mean:>12.1f}%  {gain:>+12.1f}pp  "
              f"{'YES'+marker if crossed else 'no':>10}")

        results[dwell_assumption] = {
            "baseline_fadr": b_mean,
            "ours_fadr":     ours_mean,
            "gain":          gain,
            "crossed":       crossed,
        }

    # Find crossover point
    crossovers = [d for d, r in results.items() if r["crossed"]]
    if crossovers:
        print(f"\n  CROSSOVER: system stops helping when baseline dwell > {min(crossovers):.1f} min")
        print(f"  (i.e., if the company already plans at >{min(crossovers):.1f}min dwell, gain ~0)")
    else:
        print(f"\n  NO CROSSOVER in 1–8min range: system always beats 2-min baseline.")
        print(f"  Gain is robust to dwell assumption — system valuable across all realistic baselines.")

    print(f"\n  Fixed: actual dwell N({DWELL_MEAN},{DWELL_STD}), ours P75={DWELL_P75}min")
    print(f"  Ours FADR (constant): {ours_mean:.1f}%")

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-travel",      action="store_true")
    parser.add_argument("--skip-cascade",     action="store_true")
    parser.add_argument("--skip-sensitivity", action="store_true")
    parser.add_argument("--n-cascade",  type=int, default=30)
    parser.add_argument("--n-travel",   type=int, default=50)
    parser.add_argument("--n-sweep",    type=int, default=20)
    parser.add_argument("--n-boot",     type=int, default=N_BOOT)
    args = parser.parse_args()

    print(f"\nSTATISTICAL VALIDATION SUITE")
    print(f"Bootstrap iterations: {args.n_boot}  |  Seed: {SEED}")

    cascade_result     = None
    travel_result      = None
    sensitivity_result = None

    if not args.skip_cascade:
        cascade_result = run_cascade_bootstrap(n_routes=args.n_cascade)

    if not args.skip_travel:
        travel_result = run_travel_bootstrap(n_routes=args.n_travel)

    if not args.skip_sensitivity:
        sensitivity_result = run_sensitivity_sweep(n_routes=args.n_sweep)

    print(f"\n{'='*62}")
    print(f"  FINAL DEFENSIBLE CLAIMS")
    print(f"{'='*62}")
    if cascade_result:
        cg = cascade_result
        print(f"  Cascade FADR gain:     +{cg['mean']:.1f}pp "
              f"[95% CI: {cg['lo']:.1f}–{cg['hi']:.1f}pp, p={cg['p']:.1e}]")
    if travel_result:
        tr = travel_result
        print(f"  Zone-NN travel gain:   {tr['nn']['mean']:+.1f}% "
              f"[95% CI: {tr['nn']['lo']:.1f}–{tr['nn']['hi']:.1f}%, p={tr['nn']['p']:.1e}]")
        print(f"  PyVRP travel gain:     {tr['pyvrp']['mean']:+.1f}% "
              f"[95% CI: {tr['pyvrp']['lo']:.1f}–{tr['pyvrp']['hi']:.1f}%, p={tr['pyvrp']['p']:.1e}]")
    if sensitivity_result:
        crossovers = [d for d, r in sensitivity_result.items() if r["crossed"]]
        if crossovers:
            print(f"  Sensitivity crossover: baseline dwell > {min(crossovers):.1f}min")
        else:
            print(f"  Sensitivity: robust — no crossover across 1–8min baseline range")
