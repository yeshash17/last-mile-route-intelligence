"""
models/failure_predictor.py

Predicts P(delivery fails) for each stop BEFORE the driver leaves.

Model: LightGBM classifier trained on historical delivery outcomes.
Output: { failure_probability, risk_level, recommended_action }

The recommended_action is the DI layer — it converts a probability
into a concrete operational decision: attempt / pre-call / redirect.
"""

import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

from config import HIGH_RISK_THRESHOLD, MEDIUM_RISK_THRESHOLD, MODEL_DIR


# ── Feature list ─────────────────────────────────────────────────────────────
# Must match engineer_stop_features() output in data/features.py
FEATURES = [
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "is_peak_hour",
    "building_type",
    "has_access_code",
    "has_safe_place",
    "package_weight_kg",
    "is_heavy",
    "n_previous_failed_attempts",
    "historical_success_rate",
    "hour_success_rate",
    "avg_dwell_time_mins",
    "n_past_deliveries",
]


class FailurePredictor:
    """
    Train, save, load, and predict with the failure probability model.

    Usage (training):
        predictor = FailurePredictor()
        predictor.train(df)                # df has FEATURES + 'failed' column
        predictor.save()

    Usage (inference):
        predictor = FailurePredictor.load()
        result = predictor.predict(stop_features_dict)
    """

    MODEL_FILENAME = "failure_predictor.joblib"

    def __init__(self):
        self.model = None

    # ── Training ─────────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame, target_col: str = "failed") -> "FailurePredictor":
        """
        Train on historical delivery data.

        df must contain all columns in FEATURES plus target_col (1=failed, 0=success).
        """
        X = df[FEATURES].copy()
        y = df[target_col].astype(int)

        # Typical imbalance: ~15% failures. Tell LightGBM to weight them up.
        pos_rate = y.mean()
        scale_pos_weight = (1 - pos_rate) / pos_rate

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.model = lgb.LGBMClassifier(
            n_estimators      = 600,
            learning_rate     = 0.04,
            num_leaves        = 31,
            max_depth         = -1,
            min_child_samples = 20,
            scale_pos_weight  = scale_pos_weight,
            random_state      = 42,
            n_jobs            = -1,
        )

        self.model.fit(
            X_train, y_train,
            eval_set = [(X_val, y_val)],
            callbacks = [
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=100),
            ],
        )

        # Evaluation
        val_probs = self.model.predict_proba(X_val)[:, 1]
        val_preds = (val_probs > 0.5).astype(int)
        auc = roc_auc_score(y_val, val_probs)

        print(f"\nValidation AUC: {auc:.4f}")
        print(classification_report(y_val, val_preds, target_names=["Success", "Fail"]))

        # Feature importance (top 5)
        importances = pd.Series(
            self.model.feature_importances_, index=FEATURES
        ).sort_values(ascending=False)
        print("\nTop 5 features:")
        print(importances.head(5).to_string())

        return self

    # ── Inference ────────────────────────────────────────────────────────────

    def predict(self, stop_features: dict) -> dict:
        """
        Score a single stop. Returns a dict with:
            failure_probability : float
            risk_level          : 'low' | 'medium' | 'high'
            recommended_action  : 'attempt' | 'pre_call' | 'redirect_locker'
            explanation         : human-readable reason
        """
        if self.model is None:
            raise RuntimeError("Model not trained or loaded. Call train() or load() first.")

        X = pd.DataFrame([stop_features])[FEATURES].fillna(0)
        prob_fail = float(self.model.predict_proba(X)[0][1])

        if prob_fail >= HIGH_RISK_THRESHOLD:
            risk_level = "high"
            action     = "redirect_locker"
            reason     = (
                f"Failure probability {prob_fail:.0%} exceeds {HIGH_RISK_THRESHOLD:.0%}. "
                "Recommend locker redirect or reschedule before dispatch."
            )
        elif prob_fail >= MEDIUM_RISK_THRESHOLD:
            risk_level = "medium"
            action     = "pre_call"
            reason     = (
                f"Failure probability {prob_fail:.0%}. "
                "Dispatcher should call customer to confirm availability."
            )
        else:
            risk_level = "low"
            action     = "attempt"
            reason     = f"Failure probability {prob_fail:.0%}. Proceed as normal."

        return {
            "failure_probability": round(prob_fail, 3),
            "risk_level":          risk_level,
            "recommended_action":  action,
            "explanation":         reason,
        }

    def predict_batch(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Score all stops in a DataFrame. Returns df with risk columns added."""
        X = features_df[FEATURES].fillna(0)
        probs = self.model.predict_proba(X)[:, 1]
        results = features_df.copy()
        results["failure_probability"] = probs.round(3)
        results["risk_level"] = pd.cut(
            probs,
            bins   = [0, MEDIUM_RISK_THRESHOLD, HIGH_RISK_THRESHOLD, 1.0],
            labels = ["low", "medium", "high"],
        )
        return results

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, directory: str = None) -> str:
        path = os.path.join(directory or MODEL_DIR, self.MODEL_FILENAME)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"Model saved → {path}")
        return path

    @classmethod
    def load(cls, directory: str = None) -> "FailurePredictor":
        path = os.path.join(directory or MODEL_DIR, cls.MODEL_FILENAME)
        instance = cls()
        instance.model = joblib.load(path)
        print(f"Model loaded <- {path}")
        return instance


# ── Quick test with synthetic data ───────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n   = 5000

    df = pd.DataFrame({
        "hour_of_day":                rng.integers(6, 21, n),
        "day_of_week":                rng.integers(0, 7,  n),
        "is_weekend":                 rng.integers(0, 2,  n),
        "is_peak_hour":               rng.integers(0, 2,  n),
        "building_type":              rng.integers(0, 4,  n),
        "has_access_code":            rng.integers(0, 2,  n),
        "has_safe_place":             rng.integers(0, 2,  n),
        "package_weight_kg":          rng.uniform(0.5, 30, n),
        "is_heavy":                   rng.integers(0, 2,  n),
        "n_previous_failed_attempts": rng.integers(0, 3,  n),
        "historical_success_rate":    rng.uniform(0.6, 1.0, n),
        "hour_success_rate":          rng.uniform(0.6, 1.0, n),
        "avg_dwell_time_mins":        rng.uniform(2, 8, n),
        "n_past_deliveries":          rng.integers(0, 50, n),
    })

    # Synthetic label: flats with no access code at peak hour more likely to fail
    prob_fail = (
        0.05
        + 0.20 * (df["building_type"] == 1)
        + 0.15 * (df["has_access_code"] == 0)
        + 0.10 * (df["is_peak_hour"] == 1)
        + 0.05 * (df["n_previous_failed_attempts"] > 0)
        - 0.10 * df["historical_success_rate"]
    ).clip(0.02, 0.95)

    df["failed"] = (rng.uniform(0, 1, n) < prob_fail).astype(int)

    print(f"Failure rate in synthetic data: {df['failed'].mean():.1%}")

    predictor = FailurePredictor()
    predictor.train(df)

    # Score a sample stop
    sample = {k: df[k].iloc[0] for k in FEATURES}
    result = predictor.predict(sample)
    print(f"\nSample prediction: {result}")
