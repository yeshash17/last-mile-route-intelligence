"""
models/eta_predictor.py

Predicts dwell time (minutes) per stop — time the driver spends at the door.

Model : XGBoost regressor trained on historical delivery records.
Output: dwell_mins (float) fed into the dwell_mins field in vrp_solver.py.

Why dwell time matters:
  A flat on the 4th floor without an access code during peak hour takes
  ~7 min. A business parcel drop takes ~2 min. Getting this wrong shifts
  all downstream ETAs and can cause time-window violations.
"""

import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from config import MODEL_DIR


FEATURES = [
    "building_type",
    "floor_number",
    "has_access_code",
    "has_safe_place",
    "package_weight_kg",
    "is_heavy",
    "hour_of_day",
    "is_peak_hour",
    "day_of_week",
    "is_weekend",
    "n_previous_failed_attempts",
    "avg_dwell_time_mins",        # historical mean dwell at this address
]

_MIN_DWELL = 1.0
_MAX_DWELL = 30.0


class ETAPredictor:
    """
    Train, save, load, and predict dwell time per stop.

    Training:
        predictor = ETAPredictor()
        predictor.train(df)   # df has FEATURES + 'actual_dwell_mins'
        predictor.save()

    Inference:
        predictor = ETAPredictor.load()
        dwell_mins = predictor.predict(stop_features_dict)
    """

    MODEL_FILENAME = "eta_predictor.joblib"

    def __init__(self):
        self.model = None

    # ── Training ─────────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame, target_col: str = "actual_dwell_mins") -> "ETAPredictor":
        """
        Train on historical delivery data.

        df must contain all FEATURES columns plus target_col (float, minutes).
        """
        X = df[FEATURES].copy()
        y = df[target_col].clip(_MIN_DWELL, _MAX_DWELL)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model = xgb.XGBRegressor(
            n_estimators     = 400,
            learning_rate    = 0.05,
            max_depth        = 5,
            subsample        = 0.8,
            colsample_bytree = 0.8,
            min_child_weight = 5,
            random_state     = 42,
            n_jobs           = -1,
            verbosity        = 0,
        )
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        preds = self.model.predict(X_val).clip(_MIN_DWELL, _MAX_DWELL)
        mae   = mean_absolute_error(y_val, preds)
        r2    = r2_score(y_val, preds)

        print(f"\nETA Predictor — Val MAE: {mae:.2f} min  |  R²: {r2:.4f}")

        importances = pd.Series(
            self.model.feature_importances_, index=FEATURES
        ).sort_values(ascending=False)
        print("Top 5 features:")
        print(importances.head(5).to_string())

        return self

    # ── Inference ────────────────────────────────────────────────────────────

    def predict(self, stop_features: dict) -> float:
        """
        Predict dwell time for a single stop.

        Missing features default to 0 (safe for all boolean/count fields).
        Returns float in range [1.0, 30.0] minutes.
        """
        if self.model is None:
            raise RuntimeError("Model not trained or loaded. Call train() or load() first.")

        row = {feat: stop_features.get(feat, 0) for feat in FEATURES}
        X   = pd.DataFrame([row])
        raw = float(self.model.predict(X)[0])
        return round(max(_MIN_DWELL, min(_MAX_DWELL, raw)), 1)

    def predict_batch(self, features_df: pd.DataFrame) -> np.ndarray:
        """Score all stops in a DataFrame. Returns array of floats (minutes)."""
        X   = features_df.reindex(columns=FEATURES).fillna(0)
        raw = self.model.predict(X)
        return np.clip(raw, _MIN_DWELL, _MAX_DWELL).round(1)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, directory: str = None) -> str:
        path = os.path.join(directory or MODEL_DIR, self.MODEL_FILENAME)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"ETA model saved → {path}")
        return path

    @classmethod
    def load(cls, directory: str = None) -> "ETAPredictor":
        path = os.path.join(directory or MODEL_DIR, cls.MODEL_FILENAME)
        instance = cls()
        instance.model = joblib.load(path)
        print(f"ETA model loaded ← {path}")
        return instance


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n   = 5_000

    weights = rng.uniform(0.5, 30, n)
    hours   = rng.integers(6, 21, n)

    df = pd.DataFrame({
        "building_type":              rng.integers(0, 5, n),
        "floor_number":               rng.integers(0, 10, n),
        "has_access_code":            rng.integers(0, 2, n),
        "has_safe_place":             rng.integers(0, 2, n),
        "package_weight_kg":          weights,
        "is_heavy":                   (weights > 25).astype(int),
        "hour_of_day":                hours,
        "is_peak_hour":               ((hours >= 17) & (hours < 20)).astype(int),
        "day_of_week":                rng.integers(0, 7, n),
        "is_weekend":                 rng.integers(0, 2, n),
        "n_previous_failed_attempts": rng.integers(0, 3, n),
        "avg_dwell_time_mins":        rng.uniform(2, 8, n),
    })

    # Synthetic target — encodes domain knowledge
    df["actual_dwell_mins"] = (
        2.0
        + 1.5 * (df["building_type"] == 1)     # flat
        + 0.5 * df["floor_number"]              # each floor adds time
        - 0.5 * df["has_access_code"]           # access code = quicker
        + 0.1 * df["package_weight_kg"]         # heavier = slower
        + 0.8 * df["is_peak_hour"]              # customer slower to answer
        + rng.normal(0, 0.5, n)
    ).clip(_MIN_DWELL, _MAX_DWELL)

    predictor = ETAPredictor()
    predictor.train(df)

    sample = {feat: df[feat].iloc[0] for feat in FEATURES}
    print(f"\nSample stop predicted dwell: {predictor.predict(sample)} min")
    print(f"Actual dwell:               {df['actual_dwell_mins'].iloc[0]:.1f} min")
