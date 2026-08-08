"""
feedback/collector.py

Records what ACTUALLY happened after each delivery attempt.
This is what closes the DI feedback loop — every outcome becomes
training data that makes tomorrow's predictions smarter.

Called by the driver app when each stop is completed.
"""

import sqlite3
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Literal
from config import DATABASE_URL


@dataclass
class DeliveryOutcome:
    stop_id:              str
    address:              str
    driver_id:            str
    planned_arrival:      str           # ISO datetime
    actual_arrival:       str           # ISO datetime
    success:              bool          # True = delivered, False = failed
    fail_reason:          str = ""      # "no_answer" | "wrong_address" | "access" | ""
    driver_followed_route: bool = True  # did driver use the recommended sequence?
    actual_dwell_mins:    float = 3.0   # time spent at stop
    predicted_failure_prob: float = 0.0 # what the model predicted beforehand


def record_outcome(outcome: DeliveryOutcome, db_path: str = "route_intelligence.db"):
    """Append one delivery outcome to the SQLite feedback table."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS delivery_outcomes (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            stop_id                 TEXT,
            address                 TEXT,
            driver_id               TEXT,
            planned_arrival         TEXT,
            actual_arrival          TEXT,
            success                 INTEGER,
            fail_reason             TEXT,
            driver_followed_route   INTEGER,
            actual_dwell_mins       REAL,
            predicted_failure_prob  REAL,
            logged_at               TEXT
        )
    """)
    row = asdict(outcome)
    row["success"]               = int(outcome.success)
    row["driver_followed_route"] = int(outcome.driver_followed_route)
    row["logged_at"]             = datetime.utcnow().isoformat()
    conn.execute(
        f"INSERT INTO delivery_outcomes ({','.join(row.keys())}) VALUES ({','.join('?'*len(row))})",
        list(row.values()),
    )
    conn.commit()
    conn.close()


def load_outcomes_for_training(db_path: str = "route_intelligence.db") -> pd.DataFrame:
    """
    Pull all recorded outcomes as a DataFrame ready for model retraining.
    Adds derived columns used as ML features.
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM delivery_outcomes", conn)
    conn.close()

    if df.empty:
        return df

    df["planned_arrival"] = pd.to_datetime(df["planned_arrival"])
    df["actual_arrival"]  = pd.to_datetime(df["actual_arrival"])

    # Derived features for retraining
    df["hour_of_day"]     = df["planned_arrival"].dt.hour
    df["day_of_week"]     = df["planned_arrival"].dt.weekday
    df["delay_mins"]      = (df["actual_arrival"] - df["planned_arrival"]).dt.total_seconds() / 60
    df["failed"]          = 1 - df["success"].astype(int)

    return df


def calibration_report(db_path: str = "route_intelligence.db") -> dict:
    """
    Compare predicted failure probabilities against actual outcomes.
    Tells you how well the model is calibrated.
    """
    df = load_outcomes_for_training(db_path)
    if df.empty or "predicted_failure_prob" not in df.columns:
        return {"error": "No data available yet."}

    bins = pd.cut(df["predicted_failure_prob"], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    report = df.groupby(bins)["failed"].agg(["mean", "count"]).reset_index()
    report.columns = ["predicted_prob_bucket", "actual_fail_rate", "n_deliveries"]
    return report.to_dict(orient="records")
