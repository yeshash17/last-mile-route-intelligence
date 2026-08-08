"""
models/trainer.py

Trains and saves both ML models from historical delivery data.

Data sources (in priority order):
  1. Real CSV (LaDe or your own export)  — pass --data path/to/file.csv
  2. Synthetic data (default)            — 10k records, no dataset required

After running, ./saved_models/ will contain:
  failure_predictor.joblib
  eta_predictor.joblib

Run:
    python -m models.trainer
    python -m models.trainer --data path/to/deliveries.csv --n-synthetic 20000
"""

import argparse
import os
import pandas as pd
import numpy as np

from models.failure_predictor import FailurePredictor
from models.eta_predictor import ETAPredictor
from config import MODEL_DIR


# ── LaDe column mapping ───────────────────────────────────────────────────────
# LaDe (Cainiao/Alibaba) dataset column names → internal names.
# Adjust if your CSV uses different column names.

LADE_COLUMN_MAP = {
    "lng":         "lon",
    "courier_id":  "driver_id",
    "finish_time": "delivered_at",
    "accept_time": "accepted_at",
}


def load_lade_csv(csv_path: str) -> pd.DataFrame:
    """
    Load a LaDe-format delivery CSV and map columns to internal feature names.

    Derives the 'failed' label from task_type (1 = delivered, others = failed).
    Derives 'actual_dwell_mins' from finish_time - accept_time when available.
    """
    df = pd.read_csv(csv_path)
    df = df.rename(columns=LADE_COLUMN_MAP)

    # ── Failure label ─────────────────────────────────────────────────────────
    if "failed" not in df.columns:
        if "task_type" in df.columns:
            df["failed"] = (df["task_type"] != 1).astype(int)
        else:
            raise ValueError(
                "CSV must contain 'failed' (0/1) or 'task_type' column. "
                "See SESSION_HANDOFF.md for LaDe column mapping."
            )

    # ── Dwell time ────────────────────────────────────────────────────────────
    if "actual_dwell_mins" not in df.columns:
        if "delivered_at" in df.columns and "accepted_at" in df.columns:
            df["delivered_at"] = pd.to_datetime(df["delivered_at"])
            df["accepted_at"]  = pd.to_datetime(df["accepted_at"])
            df["actual_dwell_mins"] = (
                (df["delivered_at"] - df["accepted_at"]).dt.total_seconds() / 60
            ).clip(1, 30)
        else:
            df["actual_dwell_mins"] = 3.0   # neutral default

    # ── Fill missing feature columns with safe defaults ───────────────────────
    defaults = {
        "building_type":              4,     # "other"
        "floor_number":               0,
        "has_access_code":            0,
        "has_safe_place":             0,
        "package_weight_kg":          2.0,
        "is_heavy":                   0,
        "hour_of_day":                10,
        "is_peak_hour":               0,
        "day_of_week":                1,
        "is_weekend":                 0,
        "n_previous_failed_attempts": 0,
        "historical_success_rate":    0.85,
        "hour_success_rate":          0.85,
        "avg_dwell_time_mins":        3.0,
        "n_past_deliveries":          0,
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val

    return df


def build_synthetic_dataset(n: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic delivery dataset for smoke-testing.

    Domain knowledge encoded:
      - Flats without access codes fail more at peak hours
      - Heavy packages on high floors have longer dwell times
      - Low historical success rate is the strongest failure predictor
    """
    rng = np.random.default_rng(seed)

    building_type   = rng.integers(0, 5, n)
    floor_number    = rng.integers(0, 10, n)
    has_access_code = rng.integers(0, 2, n)
    has_safe_place  = rng.integers(0, 2, n)
    weight          = rng.uniform(0.5, 30, n)
    hour_of_day     = rng.integers(6, 21, n)
    day_of_week     = rng.integers(0, 7, n)
    is_weekend      = (day_of_week >= 5).astype(int)
    is_peak_hour    = ((hour_of_day >= 17) & (hour_of_day < 20)).astype(int)
    n_prev_failed   = rng.integers(0, 4, n)
    hist_success    = rng.uniform(0.5, 1.0, n)
    hour_success    = (hist_success + rng.normal(0, 0.05, n)).clip(0, 1)
    avg_dwell       = rng.uniform(2, 8, n)
    n_past          = rng.integers(0, 100, n)

    # Failure probability (domain-encoded)
    p_fail = (
        0.05
        + 0.20 * (building_type == 1)     # flats
        + 0.15 * (has_access_code == 0)
        + 0.10 * is_peak_hour
        + 0.08 * (n_prev_failed > 0)
        - 0.15 * hist_success
    ).clip(0.02, 0.95)

    failed = (rng.uniform(0, 1, n) < p_fail).astype(int)

    # Actual dwell time (domain-encoded)
    actual_dwell = (
        2.0
        + 1.5 * (building_type == 1)
        + 0.5 * floor_number
        - 0.5 * has_access_code
        + 0.1 * weight
        + 0.8 * is_peak_hour
        + rng.normal(0, 0.5, n)
    ).clip(1.0, 30.0)

    return pd.DataFrame({
        "building_type":              building_type,
        "floor_number":               floor_number,
        "has_access_code":            has_access_code,
        "has_safe_place":             has_safe_place,
        "package_weight_kg":          weight,
        "is_heavy":                   (weight > 25).astype(int),
        "hour_of_day":                hour_of_day,
        "is_peak_hour":               is_peak_hour,
        "day_of_week":                day_of_week,
        "is_weekend":                 is_weekend,
        "n_previous_failed_attempts": n_prev_failed,
        "historical_success_rate":    hist_success,
        "hour_success_rate":          hour_success,
        "avg_dwell_time_mins":        avg_dwell,
        "n_past_deliveries":          n_past,
        "failed":                     failed,
        "actual_dwell_mins":          actual_dwell,
    })


def train_all(df: pd.DataFrame) -> None:
    """Train and save both models to MODEL_DIR."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 60)
    print("TRAINING FAILURE PREDICTOR  (LightGBM classifier)")
    print("=" * 60)
    failure_predictor = FailurePredictor()
    failure_predictor.train(df)
    failure_predictor.save(MODEL_DIR)

    print("\n" + "=" * 60)
    print("TRAINING ETA PREDICTOR  (XGBoost regressor)")
    print("=" * 60)
    eta_predictor = ETAPredictor()
    eta_predictor.train(df)
    eta_predictor.save(MODEL_DIR)

    print(f"\n✓ Both models saved to {MODEL_DIR}/")
    print("  Start the API with:  uvicorn api.main:app --reload")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train route intelligence ML models."
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to delivery CSV (LaDe format or internal). "
             "Omit to use synthetic data.",
    )
    parser.add_argument(
        "--n-synthetic", type=int, default=10_000,
        help="Synthetic sample count when --data is not provided (default: 10000).",
    )
    args = parser.parse_args()

    if args.data:
        print(f"Loading data from {args.data} ...")
        df = load_lade_csv(args.data)
        print(f"Loaded {len(df):,} records.")
    else:
        print(f"No --data provided. Generating {args.n_synthetic:,} synthetic records ...")
        df = build_synthetic_dataset(args.n_synthetic)
        print(
            f"Failure rate: {df['failed'].mean():.1%}  |  "
            f"Mean dwell: {df['actual_dwell_mins'].mean():.1f} min"
        )

    train_all(df)


if __name__ == "__main__":
    main()
