"""
models/driver_profile.py

Driver familiarity and territory persistence model.

Three mechanisms (evidence from LKH-AMZ paper and LaDe analysis):
  1. Speed multiplier   — 15% faster travel in zones with >= 10 visits
  2. Dwell reduction    — 10% less service time in buildings visited >= 5x
  3. Territory cost     — soft preference to revisit same stops (ConVRP signal)

Break-even:
  Zone familiarity:     10 zone visits  (same as L3 service time model)
  Building familiarity: 5  AOI visits

Usage:
    profile = DriverProfile.from_history(driver_id, history_df)
    mat = profile.apply_to_matrix(time_matrix, nodes)
    dwell = profile.adjusted_dwell(stop)
"""

from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants (evidence-based) ────────────────────────────────────────────────
ZONE_FAMILIARITY_THRESHOLD  = 10    # zone visits to unlock speed bonus
AOI_FAMILIARITY_THRESHOLD   = 5     # AOI visits to unlock dwell reduction
SPEED_BONUS_FAMILIAR        = 0.15  # 15% faster travel in familiar zones
DWELL_BONUS_FAMILIAR        = 0.10  # 10% less dwell in known buildings
TERRITORY_COST_REDUCTION    = 0.05  # 5% cost reduction per familiar stop (ConVRP)


@dataclass
class DriverProfile:
    driver_id:        str
    zone_visits:      dict = field(default_factory=dict)   # zone_id  → int
    aoi_visits:       dict = field(default_factory=dict)   # aoi_id   → int
    total_deliveries: int  = 0

    # ── Computed properties ───────────────────────────────────────────────────

    @property
    def familiar_zones(self) -> set:
        return {z for z, n in self.zone_visits.items() if n >= ZONE_FAMILIARITY_THRESHOLD}

    @property
    def familiar_aois(self) -> set:
        return {a for a, n in self.aoi_visits.items() if n >= AOI_FAMILIARITY_THRESHOLD}

    def zone_familiarity(self, zone_id: str) -> float:
        """0.0 = unknown, 1.0 = fully familiar (>= threshold)."""
        n = self.zone_visits.get(zone_id, 0)
        return min(1.0, n / ZONE_FAMILIARITY_THRESHOLD)

    def aoi_familiarity(self, aoi_id) -> float:
        n = self.aoi_visits.get(str(aoi_id), 0)
        return min(1.0, n / AOI_FAMILIARITY_THRESHOLD)

    # ── Core adjustments ─────────────────────────────────────────────────────

    def apply_to_matrix(self, mat: np.ndarray, nodes: list) -> np.ndarray:
        """
        Scale travel times down in familiar zones.
        nodes[0] = None (depot); nodes[1..] = Stop objects with .zone_id.

        Familiar zone → travel time × (1 - SPEED_BONUS).
        Effect: PyVRP naturally routes more stops through familiar territory.
        """
        if not self.familiar_zones:
            return mat   # no-op if driver has no history

        out = mat.copy().astype(float)
        n = len(nodes)
        for i in range(n):
            ni = nodes[i]
            if ni is None:
                continue
            if ni.zone_id in self.familiar_zones:
                # Scale all outgoing edges from familiar-zone nodes
                for j in range(n):
                    if i != j:
                        out[i, j] *= (1.0 - SPEED_BONUS_FAMILIAR)
        logger.debug(
            "Driver %s: speed bonus applied to %d familiar zones",
            self.driver_id, len(self.familiar_zones)
        )
        return out

    def adjusted_dwell(self, stop) -> float:
        """
        Reduce dwell time for buildings the driver has visited before.
        Driver knows the lift, the door code, where to leave parcels.
        """
        fam = self.aoi_familiarity(getattr(stop, "address", ""))
        reduction = fam * DWELL_BONUS_FAMILIAR
        return round(stop.dwell_mins * (1.0 - reduction), 1)

    def territory_cost_reduction(self, stop) -> float:
        """
        Fractional cost reduction for ConVRP territory persistence.
        Familiar stop → lower effective cost → solver prefers revisiting same territory.
        Returns a multiplier (0.95 for fully familiar, 1.0 for unknown).
        """
        fam = max(
            self.zone_familiarity(getattr(stop, "zone_id", "")),
            self.aoi_familiarity(getattr(stop, "address", "")),
        )
        return 1.0 - fam * TERRITORY_COST_REDUCTION

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        return {
            "driver_id":          self.driver_id,
            "total_deliveries":   self.total_deliveries,
            "familiar_zones":     len(self.familiar_zones),
            "familiar_aois":      len(self.familiar_aois),
            "speed_bonus_active": len(self.familiar_zones) > 0,
            "dwell_bonus_active": len(self.familiar_aois) > 0,
            "zone_breakdown":     {z: n for z, n in sorted(
                                   self.zone_visits.items(), key=lambda x: -x[1])[:10]},
        }

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_history(cls, driver_id: str, history: pd.DataFrame) -> "DriverProfile":
        """
        Build profile from a DataFrame of past deliveries.

        Required columns: aoi_id, zone_id (or aoi_id used as zone proxy).
        Optional: ds (date), delivery_dt (timestamp).

        history should be ALL past deliveries for this driver — the model
        counts cumulative visits, not rolling window.
        """
        profile = cls(driver_id=driver_id)
        profile.total_deliveries = len(history)

        if history.empty:
            return profile

        # Zone visits — use zone_id if present, else first 4 chars of aoi_id
        if "zone_id" in history.columns:
            zone_col = history["zone_id"].astype(str)
        elif "aoi_id" in history.columns:
            zone_col = history["aoi_id"].astype(str).str[:4]
        else:
            zone_col = pd.Series(["unknown"] * len(history))

        profile.zone_visits = zone_col.value_counts().to_dict()

        # AOI visits
        if "aoi_id" in history.columns:
            profile.aoi_visits = history["aoi_id"].astype(str).value_counts().to_dict()

        logger.info(
            "Driver %s: %d deliveries, %d familiar zones, %d familiar AOIs",
            driver_id,
            profile.total_deliveries,
            len(profile.familiar_zones),
            len(profile.familiar_aois),
        )
        return profile

    @classmethod
    def unknown(cls, driver_id: str = "unknown") -> "DriverProfile":
        """Empty profile — no history, no bonuses applied."""
        return cls(driver_id=driver_id)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, directory: str = "./saved_models") -> Path:
        path = Path(directory) / f"driver_{self.driver_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "driver_id":        self.driver_id,
                "zone_visits":      self.zone_visits,
                "aoi_visits":       self.aoi_visits,
                "total_deliveries": self.total_deliveries,
            }, f, indent=2)
        return path

    @classmethod
    def load(cls, driver_id: str, directory: str = "./saved_models") -> Optional["DriverProfile"]:
        path = Path(directory) / f"driver_{driver_id}.json"
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        p = cls(
            driver_id        = data["driver_id"],
            zone_visits      = data.get("zone_visits", {}),
            aoi_visits       = data.get("aoi_visits", {}),
            total_deliveries = data.get("total_deliveries", 0),
        )
        return p


# ── Profile cache (in-process, for API use) ───────────────────────────────────

_profile_cache: dict[str, DriverProfile] = {}

def get_driver_profile(driver_id: str,
                       directory: str = "./saved_models") -> DriverProfile:
    """Load from cache → disk → unknown (graceful degradation)."""
    if driver_id in _profile_cache:
        return _profile_cache[driver_id]
    profile = DriverProfile.load(driver_id, directory) or DriverProfile.unknown(driver_id)
    _profile_cache[driver_id] = profile
    return profile


def build_profiles_from_lade(df: pd.DataFrame,
                              directory: str = "./saved_models") -> dict:
    """
    Build and save a DriverProfile for every courier in a LaDe DataFrame.
    Run once after loading a city; profiles are reused by the API.

    Returns: {driver_id: DriverProfile}
    """
    profiles = {}
    for driver_id, history in df.groupby("courier_id"):
        p = DriverProfile.from_history(str(driver_id), history)
        p.save(directory)
        profiles[str(driver_id)] = p
    logger.info("Built %d driver profiles → %s", len(profiles), directory)
    return profiles


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.loader import load_lade

    city = sys.argv[1] if len(sys.argv) > 1 else "sh"
    print(f"Building driver profiles from LaDe-{city.upper()}...")
    df = load_lade(city=city)
    profiles = build_profiles_from_lade(df)

    # Show top 5 most experienced drivers
    top = sorted(profiles.values(), key=lambda p: p.total_deliveries, reverse=True)[:5]
    print(f"\nTop 5 drivers by delivery count:")
    print(f"{'Driver':<15} {'Deliveries':>12} {'Fam Zones':>10} {'Fam AOIs':>10} {'Speed Bonus':>12}")
    print("-" * 62)
    for p in top:
        s = p.summary()
        print(f"{p.driver_id:<15} {s['total_deliveries']:>12,} "
              f"{s['familiar_zones']:>10} {s['familiar_aois']:>10} "
              f"{'YES' if s['speed_bonus_active'] else 'no':>12}")
