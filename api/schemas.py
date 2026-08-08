"""
api/schemas.py — Pydantic models for request/response validation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime


# ── Request models ────────────────────────────────────────────────────────────

class StopRequest(BaseModel):
    stop_id:                    str
    address:                    str
    lat:                        float = Field(..., ge=-90,  le=90)
    lon:                        float = Field(..., ge=-180, le=180)
    time_window_open_mins:      int   = Field(default=0,   ge=0, le=1440)
    time_window_close_mins:     int   = Field(default=480, ge=0, le=1440)
    package_weight_kg:          float = Field(..., gt=0)
    building_type:              Literal["house","flat","business","locker","other"] = "other"
    has_access_code:            bool  = False
    has_safe_place:             bool  = False
    n_previous_failed_attempts: int   = Field(default=0, ge=0)
    historical_success_rate:    float = Field(default=0.85, ge=0, le=1)
    planned_arrival_dt:         Optional[datetime] = None
    aoi_type:                   int   = Field(default=1, ge=0)   # 1=residential, 2=office
    zone_id:                    str   = ""                        # for zone-coherence routing

    model_config = {"json_schema_extra": {
        "example": {
            "stop_id":               "S-001",
            "address":               "42 Baker Street, London W1U 6AJ",
            "lat":                   51.5238,
            "lon":                   -0.1585,
            "time_window_open_mins": 120,
            "time_window_close_mins":300,
            "package_weight_kg":     3.2,
            "building_type":         "flat",
            "has_access_code":       False,
            "has_safe_place":        True,
            "historical_success_rate": 0.88,
        }
    }}


class OptimiseRouteRequest(BaseModel):
    driver_id:           str
    depot_lat:           float = Field(default=51.5282, ge=-90,  le=90)   # default: Euston
    depot_lon:           float = Field(default=-0.1337, ge=-180, le=180)
    shift_start:         datetime
    shift_duration_mins: int   = Field(default=480, ge=60, le=720)
    vehicle_capacity_kg: float = Field(default=500.0, gt=0)
    stops:               List[StopRequest] = Field(..., min_length=1, max_length=200)


class RiskScoreRequest(BaseModel):
    stop_id:                    str
    building_type:              str   = "other"
    has_access_code:            bool  = False
    has_safe_place:             bool  = False
    package_weight_kg:          float = 2.0
    planned_hour:               int   = Field(default=10, ge=0, le=23)
    planned_day_of_week:        int   = Field(default=1,  ge=0, le=6)
    n_previous_failed_attempts: int   = Field(default=0,  ge=0)
    historical_success_rate:    float = Field(default=0.85, ge=0, le=1)


# ── Response models ───────────────────────────────────────────────────────────

class RiskResult(BaseModel):
    failure_probability: float
    risk_level:          Literal["low", "medium", "high"]
    recommended_action:  Literal["attempt", "pre_call", "redirect_locker"]
    explanation:         str


class RouteStepResponse(BaseModel):
    sequence:    int
    stop_id:     str
    address:     str
    arrival_min: int      # minutes from shift start
    dwell_mins:  float
    p_success:   float
    risk:        Optional[RiskResult] = None


class DeferredStop(BaseModel):
    stop_id:    str
    address:    str
    p_success:  float
    reason:     str = "High failure probability — retry tomorrow in a different time band"


class OptimiseRouteResponse(BaseModel):
    driver_id:        str
    total_stops:      int
    total_time_mins:  float
    expected_fadr:    float
    cascade_risk:     bool
    route:            List[RouteStepResponse]
    deferred_stops:   List[DeferredStop] = []
    cost_breakdown:   dict = {}
    generated_at:     datetime


class HealthResponse(BaseModel):
    status:  Literal["ok", "degraded"]
    models:  dict
    version: str = "0.2.0"
