"""
data/loader.py

Loads LaDe-D parquet files and extracts clean service-time ground truth
using the same-location consecutive delivery trick from the methodology doc.

The five data-quality checks applied (D.7):
  1. Timestamp semantics   — delivery_time is task completion; use it, not accept_time
  2. Station/hub exclusion — aoi_type 15 (2 depot hubs) and aoi_type 9 (high GPS noise)
  3. Idle-time filtering   — GPS drift > 500m means courier wasn't at the stop; drop row
  4. Stacked deliveries    — aggregate to stop level before computing gaps
  5. GPS quality check     — haversine(stop_coords, gps_coords) < 500m per row

Service-time extraction trick (methodology doc D.2):
  For consecutive deliveries at the same aoi_id:
      travel_time ≈ 0  →  gap ≈ service_time
  This gives clean ground truth without any model.

Usage:
    from data.loader import load_lade, extract_service_times

    df        = load_lade(city="sh")
    gaps_df   = extract_service_times(df)   # same-location gaps only
    full_gaps = extract_all_gaps(df)        # all inter-stop gaps (travel + service)
"""

import math
import logging
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).parent.parent.parent   # Last Mile Route Intelligence Engine/
LADE_DIR   = _REPO_ROOT / "data" / "lade" / "data"

CITY_FILES = {
    "sh":  "delivery_sh-00000-of-00001-ad9a4b1d79823540.parquet",   # Shanghai  1.48M
    "hz":  "delivery_hz-00000-of-00001-8090c86f64781f71.parquet",   # Hangzhou  1.86M
    "cq":  "delivery_cq-00000-of-00001-465887add76aeabc.parquet",   # Chongqing  931K
    "yt":  "delivery_yt-00000-of-00001-cc85c1fcb1d10955.parquet",   # Yantai     206K
    "jl":  "delivery_jl-00000-of-00001-a4fbefe3c368583c.parquet",   # Jilin       31K
}

# ── Quality-filter constants ──────────────────────────────────────────────────

# aoi_type 15 = 2 unique depot hubs (664 records/location) — exclude
# aoi_type 9  = high GPS noise (mean drift 0.030° ≈ 3km)   — exclude
EXCLUDED_AOI_TYPES = {9, 15}

GPS_DRIFT_MAX_KM     = 0.5    # courier must be within 500m of recorded stop
SAME_LOC_TRAVEL_KM   = 0.05   # consecutive stops considered "same location" if <50m apart
MAX_SERVICE_MINS     = 60.0   # cap: 1h gap at same location = not service time (lunch/break)
MIN_SERVICE_MINS     = 0.5    # floor: <30s gaps are scan artifacts


# ── Haversine ─────────────────────────────────────────────────────────────────

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    la1, lo1, la2, lo2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = la2 - la1, lo2 - lo1
    a = math.sin(dlat / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


# ── Loading ───────────────────────────────────────────────────────────────────

def load_lade(
    city: str = "sh",
    data_dir: Optional[Path] = None,
    sample_n: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load a LaDe-D city parquet and apply all five quality checks.

    Parameters
    ----------
    city     : 'sh' | 'hz' | 'cq' | 'yt' | 'jl'
    data_dir : override default path (useful for testing)
    sample_n : if set, return a random sample of this many rows (for fast dev)

    Returns
    -------
    DataFrame with parsed timestamps, GPS drift column, and quality flags.
    Only rows passing all quality checks are returned.
    """
    base = Path(data_dir) if data_dir else LADE_DIR
    fname = CITY_FILES.get(city)
    if not fname:
        raise ValueError(f"Unknown city '{city}'. Valid: {list(CITY_FILES)}")

    path = base / fname
    if not path.exists():
        raise FileNotFoundError(
            f"LaDe file not found: {path}\n"
            f"Download with: hf download Cainiao-AI/LaDe-D --repo-type dataset "
            f"--local-dir <your-data-dir>/lade"
        )

    logger.info("Loading LaDe-%s from %s ...", city.upper(), path)
    df = pd.read_parquet(path)
    n_raw = len(df)

    # ── Quality check 1: timestamp parsing ───────────────────────────────────
    # delivery_time format: "MM-DD HH:MM:SS", ds = MMDD integer
    # accept_time is station pickup time — NOT used for service-time gaps.
    df["delivery_dt"] = pd.to_datetime(
        "2024-" + df["delivery_time"].str.strip(),
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )
    bad_ts = df["delivery_dt"].isna().sum()
    if bad_ts > 0:
        logger.warning("Dropping %d rows with unparseable delivery_time.", bad_ts)
        df = df[df["delivery_dt"].notna()].copy()

    # ── Quality check 2: station/hub exclusion ────────────────────────────────
    df = df[~df["aoi_type"].isin(EXCLUDED_AOI_TYPES)].copy()
    logger.info(
        "After station exclusion (types %s): %d → %d rows.",
        EXCLUDED_AOI_TYPES, n_raw, len(df),
    )

    # ── Quality check 3: GPS drift filter ────────────────────────────────────
    # Vectorised haversine between stop coords (lng/lat) and courier GPS at delivery
    R = 6371.0
    la1 = np.radians(df["lat"].values)
    lo1 = np.radians(df["lng"].values)
    la2 = np.radians(df["delivery_gps_lat"].values)
    lo2 = np.radians(df["delivery_gps_lng"].values)
    dlat = la2 - la1
    dlon = lo2 - lo1
    a = np.sin(dlat / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(dlon / 2) ** 2
    df["gps_drift_km"] = R * 2 * np.arcsin(np.sqrt(a))

    n_before = len(df)
    df = df[df["gps_drift_km"] <= GPS_DRIFT_MAX_KM].copy()
    logger.info(
        "GPS drift filter (>%.0fm): dropped %d rows.",
        GPS_DRIFT_MAX_KM * 1000, n_before - len(df),
    )

    # ── Quality check 4: aggregate to stop level ──────────────────────────────
    # Multiple packages at same (courier, ds, aoi_id, delivery_time) = stacked delivery.
    # Aggregate: keep min delivery_time per group, count packages.
    df["package_count"] = 1
    df = (
        df.groupby(["courier_id", "ds", "aoi_id", "delivery_dt"], as_index=False)
        .agg(
            aoi_type       = ("aoi_type",  "first"),
            lat            = ("lat",        "first"),
            lng            = ("lng",        "first"),
            gps_drift_km   = ("gps_drift_km", "mean"),
            package_count  = ("package_count", "sum"),
        )
        .sort_values(["courier_id", "ds", "delivery_dt"])
        .reset_index(drop=True)
    )

    # ── Quality check 5: GPS quality already applied above (check 3) ──────────

    if sample_n and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)

    logger.info("Final cleaned dataset: %d stop records.", len(df))
    return df


# ── Gap extraction ────────────────────────────────────────────────────────────

def extract_service_times(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract same-location consecutive delivery gaps as service-time ground truth.

    For consecutive stops where aoi_id[i] == aoi_id[i+1] (same building),
    travel_time ≈ 0 so gap ≈ service_time. Clean ground truth with no modelling.

    Returns DataFrame with columns:
        aoi_id, aoi_type, lat, lng, package_count, hour_of_day, day_of_week,
        service_mins  ← the ground truth
    """
    rows = []

    for (courier, ds), grp in df.groupby(["courier_id", "ds"]):
        grp = grp.sort_values("delivery_dt").reset_index(drop=True)
        for i in range(len(grp) - 1):
            curr = grp.iloc[i]
            nxt  = grp.iloc[i + 1]

            # Same AOI = same-location consecutive delivery
            if curr["aoi_id"] != nxt["aoi_id"]:
                continue

            gap_mins = (nxt["delivery_dt"] - curr["delivery_dt"]).total_seconds() / 60

            if not (MIN_SERVICE_MINS <= gap_mins <= MAX_SERVICE_MINS):
                continue   # artifact or break/lunch

            rows.append({
                "aoi_id":       curr["aoi_id"],
                "aoi_type":     curr["aoi_type"],
                "lat":          curr["lat"],
                "lng":          curr["lng"],
                "package_count": nxt["package_count"],
                "hour_of_day":  nxt["delivery_dt"].hour,
                "day_of_week":  nxt["delivery_dt"].weekday(),
                "service_mins": round(gap_mins, 2),
            })

    result = pd.DataFrame(rows)
    logger.info(
        "Extracted %d same-location service-time observations.", len(result)
    )
    return result


def extract_all_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract all inter-stop gaps (travel + service time) per courier per day.

    Returns DataFrame with columns:
        courier_id, ds, from_aoi, to_aoi, aoi_type,
        gap_mins, same_location, package_count, hour_of_day
    """
    rows = []

    for (courier, ds), grp in df.groupby(["courier_id", "ds"]):
        grp = grp.sort_values("delivery_dt").reset_index(drop=True)
        for i in range(len(grp) - 1):
            curr = grp.iloc[i]
            nxt  = grp.iloc[i + 1]
            gap_mins = (nxt["delivery_dt"] - curr["delivery_dt"]).total_seconds() / 60

            if gap_mins <= 0 or gap_mins > 180:   # skip breaks / overnight
                continue

            rows.append({
                "courier_id":    courier,
                "ds":            ds,
                "from_aoi":      curr["aoi_id"],
                "to_aoi":        nxt["aoi_id"],
                "aoi_type":      nxt["aoi_type"],
                "lat":           nxt["lat"],
                "lng":           nxt["lng"],
                "gap_mins":      round(gap_mins, 2),
                "same_location": curr["aoi_id"] == nxt["aoi_id"],
                "package_count": nxt["package_count"],
                "hour_of_day":   nxt["delivery_dt"].hour,
                "day_of_week":   nxt["delivery_dt"].weekday(),
            })

    return pd.DataFrame(rows)


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print("Loading LaDe Shanghai (sample 50k rows for speed)...")
    df = load_lade(city="sh", sample_n=50_000)
    print(f"Cleaned dataset: {len(df):,} stop records\n")

    print("Extracting same-location service times...")
    st = extract_service_times(df)
    if len(st) > 0:
        print(f"Service time observations: {len(st):,}")
        print(f"Mean service time:  {st['service_mins'].mean():.1f} min")
        print(f"Median:             {st['service_mins'].median():.1f} min")
        print(f"P90:                {st['service_mins'].quantile(0.9):.1f} min")
        print(f"\nBy aoi_type:")
        print(st.groupby("aoi_type")["service_mins"].agg(["count","mean","median"]).to_string())
    else:
        print("No same-location pairs found in sample — increase sample_n.")

    print("\nExtracting all inter-stop gaps...")
    gaps = extract_all_gaps(df)
    print(f"Total gaps: {len(gaps):,}")
    print(f"Same-location gaps: {gaps['same_location'].sum():,} ({gaps['same_location'].mean():.1%})")
