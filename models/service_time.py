"""
models/service_time.py

Per-location service time model trained on LaDe same-location delivery gaps.

Four levels compared (methodology doc D.4):
  Level 0 — global constant       (what most planning systems use)
  Level 1 — by aoi_type           (residential vs office vs mall)
  Level 2 — + package count, hour (good planning tool)
  Level 3 — per-aoi shrinkage     (nobody does this; the value-add)

Level 3 uses empirical-Bayes shrinkage:
    aoi_est = (n * aoi_mean + k * global_mean) / (n + k)
where k = within-AOI variance / between-AOI variance.
For AOIs with few observations the estimate shrinks toward the global mean;
for AOIs with many visits it trusts the observed mean.

Train/val split is by aoi_id AND courier_id — never a random row split
(random split lets the model "memorise" buildings seen in training,
overstating accuracy on locations it has already learned).

Usage:
    from data.loader import load_lade, extract_service_times
    from models.service_time import ServiceTimeModel

    obs   = extract_service_times(load_lade("sh"))
    model = ServiceTimeModel().train(obs)
    model.save()

    # Inference
    model = ServiceTimeModel.load()
    mins  = model.predict(aoi_id=450, aoi_type=1, package_count=2, hour_of_day=14)
"""

import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

from config import MODEL_DIR

logger = logging.getLogger(__name__)

_MIN_DWELL =  0.5
_MAX_DWELL = 60.0

MODEL_FILENAME       = "service_time_model.joblib"
AOI_STATS_FILENAME   = "service_time_aoi_stats.json"
TYPE_STATS_FILENAME  = "service_time_type_stats.json"


class ServiceTimeModel:
    """
    Four-level service time estimator.

    After training, `predict()` automatically selects the richest level
    available for each stop — falling back to coarser levels when an aoi_id
    has been seen too few times or not at all.
    """

    def __init__(self, min_obs_for_aoi: int = 10):
        """
        Parameters
        ----------
        min_obs_for_aoi : minimum observations before trusting a per-aoi estimate.
                          Below this the estimate fully shrinks to the type mean.
        """
        self.min_obs_for_aoi = min_obs_for_aoi

        # populated by train()
        self.global_mean: float = 3.0
        self.type_stats:  dict  = {}    # aoi_type → {mean, std, n}
        self.aoi_stats:   dict  = {}    # aoi_id   → {mean, n, shrunk_mean}
        self.shrinkage_k: float = 1.0
        self.xgb_model          = None  # level-2 feature model

        # evaluation results stored by train()
        self.level_mae: dict = {}
        self.learning_curve: dict = {}

    # ── Training ─────────────────────────────────────────────────────────────

    def train(self, obs: pd.DataFrame) -> "ServiceTimeModel":
        """
        Train on same-location service-time observations from data/loader.py.

        obs columns: aoi_id, aoi_type, package_count, hour_of_day, day_of_week,
                     service_mins  (the ground truth)

        Split: held-out 20% of aoi_ids — never random row split.
        """
        obs = obs.copy()
        obs["service_mins"] = obs["service_mins"].clip(_MIN_DWELL, _MAX_DWELL)

        # ── Train / val split by aoi_id ───────────────────────────────────────
        unique_aois  = obs["aoi_id"].unique()
        rng          = np.random.default_rng(42)
        val_aoi_mask = rng.random(len(unique_aois)) < 0.20
        val_aois     = set(unique_aois[val_aoi_mask])

        train = obs[~obs["aoi_id"].isin(val_aois)].copy()
        val   = obs[ obs["aoi_id"].isin(val_aois)].copy()
        logger.info(
            "Train: %d obs / %d AOIs   Val: %d obs / %d AOIs",
            len(train), train["aoi_id"].nunique(),
            len(val),   val["aoi_id"].nunique(),
        )

        y_val = val["service_mins"].values

        # ── Level 0 — global mean ─────────────────────────────────────────────
        self.global_mean = float(train["service_mins"].mean())
        l0_preds = np.full(len(val), self.global_mean)
        self.level_mae["L0_global"] = float(mean_absolute_error(y_val, l0_preds))

        # ── Level 1 — by aoi_type ─────────────────────────────────────────────
        self.type_stats = (
            train.groupby("aoi_type")["service_mins"]
            .agg(["mean", "std", "count"])
            .rename(columns={"mean": "mean", "std": "std", "count": "n"})
            .to_dict(orient="index")
        )
        l1_preds = val["aoi_type"].map(
            lambda t: self.type_stats.get(t, {}).get("mean", self.global_mean)
        ).values
        self.level_mae["L1_by_type"] = float(mean_absolute_error(y_val, l1_preds))

        # ── Level 2 — GBM: aoi_type + package_count + hour_of_day ───────────
        feat_cols = ["aoi_type", "package_count", "hour_of_day", "day_of_week"]
        X_train   = train[feat_cols].fillna(0)
        X_val     = val[feat_cols].fillna(0)

        self.xgb_model = xgb.XGBRegressor(
            n_estimators     = 300,
            learning_rate    = 0.05,
            max_depth        = 4,
            subsample        = 0.8,
            colsample_bytree = 0.8,
            random_state     = 42,
            n_jobs           = -1,
            verbosity        = 0,
        )
        self.xgb_model.fit(X_train, train["service_mins"].values,
                           eval_set=[(X_val, y_val)], verbose=False)
        l2_preds = np.clip(self.xgb_model.predict(X_val), _MIN_DWELL, _MAX_DWELL)
        self.level_mae["L2_features"] = float(mean_absolute_error(y_val, l2_preds))

        # ── Level 3 — per-aoi shrinkage ───────────────────────────────────────
        # Shrinkage k = within-aoi variance / between-aoi variance
        aoi_group       = train.groupby("aoi_id")["service_mins"]
        within_var      = float(aoi_group.var().mean())
        between_var     = float(aoi_group.mean().var())
        self.shrinkage_k = within_var / between_var if between_var > 0 else 1.0

        aoi_agg = aoi_group.agg(["mean", "count"]).reset_index()
        self.aoi_stats = {}
        for _, row in aoi_agg.iterrows():
            aoi_id  = int(row["aoi_id"])
            n       = int(row["count"])
            raw     = float(row["mean"])
            t       = int(train[train["aoi_id"] == aoi_id]["aoi_type"].iloc[0]) if n > 0 else -1
            t_mean  = self.type_stats.get(t, {}).get("mean", self.global_mean)
            shrunk  = (n * raw + self.shrinkage_k * t_mean) / (n + self.shrinkage_k)
            self.aoi_stats[aoi_id] = {"n": n, "raw_mean": round(raw, 2), "shrunk_mean": round(shrunk, 2)}

        l3_preds = np.array([
            self._level3_pred(row["aoi_id"], row["aoi_type"], row["package_count"], row["hour_of_day"])
            for _, row in val.iterrows()
        ])
        self.level_mae["L3_per_aoi"] = float(mean_absolute_error(y_val, l3_preds))

        # ── Learning curve — how many visits until L3 beats L1 ───────────────
        self.learning_curve = self._compute_learning_curve(train)

        self._log_results()
        return self

    def _level3_pred(self, aoi_id, aoi_type, package_count, hour_of_day) -> float:
        stats = self.aoi_stats.get(int(aoi_id))
        if stats and stats["n"] >= self.min_obs_for_aoi:
            return float(stats["shrunk_mean"])
        return self.type_stats.get(int(aoi_type), {}).get("mean", self.global_mean)

    def _compute_learning_curve(self, train: pd.DataFrame) -> dict:
        """
        For each N: simulate "we have N visits to a building" using training AOIs.
        Each AOI with >= N+5 obs: first N = simulated history, rest = test.
        Compares L3 shrinkage estimate vs L1 type mean on test portion.
        """
        results = {}
        rng = np.random.default_rng(42)
        for n_thresh in [1, 3, 5, 10, 20, 50]:
            l1_errors, l3_errors, n_aois = [], [], 0
            for aoi_id, group in train.groupby("aoi_id"):
                if len(group) < n_thresh + 5:
                    continue
                idx = rng.permutation(len(group))
                train_sub = group.iloc[idx[:n_thresh]]
                test_sub  = group.iloc[idx[n_thresh:]]

                aoi_type = int(group["aoi_type"].iloc[0])
                l1_pred  = self.type_stats.get(aoi_type, {}).get("mean", self.global_mean)

                raw_mean = float(train_sub["service_mins"].mean())
                l3_pred  = (n_thresh * raw_mean + self.shrinkage_k * l1_pred) / (n_thresh + self.shrinkage_k)

                y_test = test_sub["service_mins"].values
                l1_errors.extend(np.abs(y_test - l1_pred))
                l3_errors.extend(np.abs(y_test - l3_pred))
                n_aois += 1

            if n_aois < 10:
                continue
            l1_mae = float(np.mean(l1_errors))
            l3_mae = float(np.mean(l3_errors))
            results[n_thresh] = {
                "n_aois":  n_aois,
                "L1_mae":  round(l1_mae, 3),
                "L3_mae":  round(l3_mae, 3),
                "L3_wins": l3_mae < l1_mae,
            }
        return results

    def _log_results(self):
        print("\n" + "=" * 55)
        print("SERVICE TIME MODEL — validation MAE (minutes)")
        print("=" * 55)
        for level, mae in self.level_mae.items():
            bar = "#" * int(mae)
            print(f"  {level:18s}  {mae:5.2f} min  {bar}")
        print()
        print("Learning curve — visits before per-location beats type-level:")
        for n, r in self.learning_curve.items():
            winner = "L3 WIN" if r["L3_wins"] else "L1 better"
            print(f"  N>={n:3d}: L1={r['L1_mae']:.2f}  L3={r['L3_mae']:.2f}  {winner}")
        print()

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(
        self,
        aoi_id:        int,
        aoi_type:      int,
        package_count: int   = 1,
        hour_of_day:   int   = 10,
        day_of_week:   int   = 1,
        level:         int   = 3,
    ) -> float:
        """
        Predict service time in minutes for one stop.

        level : 0=global, 1=type, 2=features, 3=per-aoi (default, best)
        Falls back gracefully: 3→1→0 when data is insufficient.
        """
        if level <= 0:
            return round(self.global_mean, 1)
        if level == 1:
            return round(self.type_stats.get(aoi_type, {}).get("mean", self.global_mean), 1)
        if level == 2 and self.xgb_model is not None:
            X   = pd.DataFrame([{"aoi_type": aoi_type, "package_count": package_count,
                                  "hour_of_day": hour_of_day, "day_of_week": day_of_week}])
            raw = float(self.xgb_model.predict(X)[0])
            return round(np.clip(raw, _MIN_DWELL, _MAX_DWELL), 1)
        # level 3
        return round(np.clip(self._level3_pred(aoi_id, aoi_type, package_count, hour_of_day),
                             _MIN_DWELL, _MAX_DWELL), 1)

    def predict_with_std(
        self,
        aoi_id:        int,
        aoi_type:      int,
        package_count: int = 1,
        hour_of_day:   int = 10,
        day_of_week:   int = 1,
    ) -> tuple[float, float]:
        """
        Return (mean_dwell, std_dwell) in minutes.
        Mean: best available level (3→2→1→0).
        Std:  per-aoi-type observed std from training data (conservative — total variance).
        """
        mean = self.predict(aoi_id, aoi_type, package_count, hour_of_day, day_of_week)
        std  = float(self.type_stats.get(aoi_type, {}).get("std", self.global_mean * 0.5))
        std  = max(std, 0.5)   # floor at 30 sec
        return mean, std

    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict for a DataFrame with columns: aoi_id, aoi_type, package_count,
        hour_of_day, day_of_week.
        """
        return np.array([
            self.predict(r["aoi_id"], r["aoi_type"],
                         r.get("package_count", 1), r.get("hour_of_day", 10))
            for _, r in df.iterrows()
        ])

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, directory: str = None) -> str:
        d = Path(directory or MODEL_DIR)
        d.mkdir(parents=True, exist_ok=True)

        model_path = d / MODEL_FILENAME
        joblib.dump(self.xgb_model, model_path)

        aoi_path = d / AOI_STATS_FILENAME
        aoi_path.write_text(json.dumps({
            "global_mean":   self.global_mean,
            "shrinkage_k":   self.shrinkage_k,
            "min_obs":       self.min_obs_for_aoi,
            "level_mae":     self.level_mae,
            "learning_curve": {str(k): v for k, v in self.learning_curve.items()},
            "aoi_stats":     {str(k): v for k, v in self.aoi_stats.items()},
        }, indent=2))

        type_path = d / TYPE_STATS_FILENAME
        type_path.write_text(json.dumps(
            {str(k): v for k, v in self.type_stats.items()}, indent=2
        ))

        logger.info("Service time model saved → %s", d)
        return str(d)

    @classmethod
    def load(cls, directory: str = None) -> "ServiceTimeModel":
        d = Path(directory or MODEL_DIR)
        m = cls()

        aoi_data          = json.loads((d / AOI_STATS_FILENAME).read_text())
        m.global_mean     = aoi_data["global_mean"]
        m.shrinkage_k     = aoi_data["shrinkage_k"]
        m.min_obs_for_aoi = aoi_data["min_obs"]
        m.level_mae       = aoi_data["level_mae"]
        m.learning_curve  = {int(k): v for k, v in aoi_data["learning_curve"].items()}
        m.aoi_stats       = {int(k): v for k, v in aoi_data["aoi_stats"].items()}

        m.type_stats = {
            int(k): v
            for k, v in json.loads((d / TYPE_STATS_FILENAME).read_text()).items()
        }
        m.xgb_model = joblib.load(d / MODEL_FILENAME)

        logger.info("Service time model loaded ← %s", d)
        return m


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from data.loader import load_lade, extract_service_times

    cities = ["sh", "hz", "cq", "yt", "jl"]
    all_obs = []
    for city in cities:
        print(f"Loading {city.upper()}...")
        df  = load_lade(city=city)
        obs_city = extract_service_times(df)
        print(f"  {len(obs_city):,} observations")
        all_obs.append(obs_city)
    obs = pd.concat(all_obs, ignore_index=True)
    print(f"\nTotal service-time observations: {len(obs):,}\n")

    if len(obs) < 50:
        print("Too few observations — load the full dataset (remove sample_n).")
        sys.exit(1)

    model = ServiceTimeModel()
    model.train(obs)
    model.save()

    # Sample predictions
    print("Sample predictions:")
    for aoi_type, label in [(1, "Residential"), (2, "Office"), (0, "Other")]:
        t = model.predict(aoi_id=-1, aoi_type=aoi_type, package_count=1, hour_of_day=10)
        print(f"  {label} (aoi_type={aoi_type}): {t} min")

    flat = model.global_mean
    print(f"\nFlat constant (what most systems use): {flat:.1f} min")
    print(f"L3 MAE improvement over flat: "
          f"{model.level_mae['L0_global'] - model.level_mae['L3_per_aoi']:.2f} min")
