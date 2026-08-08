"""
data/features.py

Transforms a raw stop record (from the delivery manifest + history DB)
into the feature vector used by both the failure predictor and ETA model.

Input:  dict or pd.Series with raw stop fields
Output: dict of engineered features
"""

import pandas as pd
import numpy as np
from datetime import datetime


BUILDING_TYPE_MAP = {
    "house":       0,
    "flat":        1,
    "business":    2,
    "locker":      3,
    "other":       4,
}


def engineer_stop_features(stop: dict, history_df: pd.DataFrame | None = None) -> dict:
    """
    Build the feature dict for a single stop at decision time.

    Parameters
    ----------
    stop : dict
        Raw stop record. Expected keys:
            address, lat, lon, building_type, has_access_code, has_safe_place,
            package_weight_kg, planned_arrival_dt (datetime or ISO string),
            n_previous_failed_attempts (int)
    history_df : pd.DataFrame | None
        Historical delivery records for this address.
        Columns: [delivery_date, hour_of_day, success (bool)]
        If None, historical features default to neutral values.

    Returns
    -------
    dict : feature vector ready for model.predict()
    """
    # --- Parse arrival time ---
    dt = stop.get("planned_arrival_dt")
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)
    elif dt is None:
        dt = datetime.now()

    hour_of_day  = dt.hour
    day_of_week  = dt.weekday()          # 0=Monday … 6=Sunday
    is_weekend   = int(day_of_week >= 5)
    is_peak_hour = int(hour_of_day in range(17, 20))  # 5–8pm busy

    # --- Address features ---
    building_code = BUILDING_TYPE_MAP.get(
        str(stop.get("building_type", "other")).lower(), 4
    )
    has_access_code = int(bool(stop.get("has_access_code", False)))
    has_safe_place  = int(bool(stop.get("has_safe_place", False)))

    # --- Package features ---
    weight_kg = float(stop.get("package_weight_kg", 1.0))
    is_heavy  = int(weight_kg > 30)

    # --- Attempt history ---
    n_prev_failed = int(stop.get("n_previous_failed_attempts", 0))

    # --- Historical success rate at this address ---
    if history_df is not None and len(history_df) > 0:
        historical_success_rate = float(history_df["success"].mean())
        avg_dwell_time_mins     = float(history_df.get("dwell_mins", pd.Series([3])).mean())
        n_past_deliveries       = len(history_df)
        # Success rate at the same hour of day
        same_hour = history_df[history_df["hour_of_day"] == hour_of_day]
        hour_success_rate = float(same_hour["success"].mean()) if len(same_hour) > 0 else historical_success_rate
    else:
        historical_success_rate = 0.85   # population-level prior
        avg_dwell_time_mins     = 3.0
        n_past_deliveries       = 0
        hour_success_rate       = 0.85

    return {
        # Temporal
        "hour_of_day":          hour_of_day,
        "day_of_week":          day_of_week,
        "is_weekend":           is_weekend,
        "is_peak_hour":         is_peak_hour,
        # Address
        "building_type":        building_code,
        "has_access_code":      has_access_code,
        "has_safe_place":       has_safe_place,
        # Package
        "package_weight_kg":    weight_kg,
        "is_heavy":             is_heavy,
        # Attempt history
        "n_previous_failed_attempts": n_prev_failed,
        # Historical intelligence
        "historical_success_rate":    historical_success_rate,
        "hour_success_rate":          hour_success_rate,
        "avg_dwell_time_mins":        avg_dwell_time_mins,
        "n_past_deliveries":          n_past_deliveries,
    }


def engineer_batch(manifest_df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features for an entire day's delivery manifest.

    Parameters
    ----------
    manifest_df : DataFrame with one row per stop
    history_df  : DataFrame with all historical records
                  must have column 'address' for joining

    Returns
    -------
    DataFrame of features, same row order as manifest_df
    """
    rows = []
    for _, stop in manifest_df.iterrows():
        addr_history = history_df[history_df["address"] == stop["address"]]
        feats = engineer_stop_features(stop.to_dict(), addr_history)
        feats["stop_id"] = stop.get("stop_id", stop.name)
        rows.append(feats)
    return pd.DataFrame(rows)


# ── Quick smoke test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_stop = {
        "address":                   "42 High Street, London",
        "lat":                       51.509865,
        "lon":                       -0.118092,
        "building_type":             "flat",
        "has_access_code":           False,
        "has_safe_place":            False,
        "package_weight_kg":         2.5,
        "planned_arrival_dt":        "2024-03-15T14:30:00",
        "n_previous_failed_attempts": 1,
    }
    feats = engineer_stop_features(sample_stop)
    for k, v in feats.items():
        print(f"  {k:35s}: {v}")
