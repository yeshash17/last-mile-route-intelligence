"""
data/distance_matrix.py

Builds an n×n travel-time matrix (minutes) for a list of (lat, lon) coordinates.

Priority:
  1. Google Maps Distance Matrix API  — real road distances + live traffic
  2. Haversine fallback               — straight-line at 25 km/h, zero API cost

Usage:
    from data.distance_matrix import build_matrix

    coords = [(51.509865, -0.118092), (51.5238, -0.1585), ...]
    matrix = build_matrix(coords)   # uses API if GOOGLE_MAPS_API_KEY is set
"""

import math
import logging
import os
import requests
from typing import List, Tuple

from config import GOOGLE_MAPS_API_KEY

logger = logging.getLogger(__name__)

# ── Backend config ────────────────────────────────────────────────────────────
# Priority: 1. OSRM self-hosted  2. Google Maps API  3. Haversine fallback
# Set OSRM_URL in .env to point at your self-hosted instance.

OSRM_URL            = os.getenv("OSRM_URL", "http://localhost:5000")
_GMAPS_URL          = "https://maps.googleapis.com/maps/api/distancematrix/json"
_URBAN_SPEED_KMH    = 25     # haversine fallback assumed speed
_BATCH_SIZE         = 10     # 10×10 = 100 elements — Google API limit per request
_OSRM_BATCH         = 100   # max coords per OSRM table request

# Per-region OSRM servers (lat_min, lat_max, lon_min, lon_max, url)
_OSRM_REGIONS = [
    (32.5, 42.0, -124.5, -114.0, os.getenv("OSRM_CA_URL",  "http://localhost:5001")),  # California
    (41.0, 47.5, -125.0, -116.5, os.getenv("OSRM_WA_URL",  "http://localhost:5002")),  # Washington
    (25.8, 36.5, -107.0,  -93.5, os.getenv("OSRM_TX_URL",  "http://localhost:5003")),  # Texas
    (36.9, 43.0,  -91.5,  -87.0, os.getenv("OSRM_IL_URL",  "http://localhost:5004")),  # Illinois
    (41.2, 47.5,  -73.5,  -69.9, os.getenv("OSRM_MA_URL",  "http://localhost:5005")),  # Massachusetts
]


def _select_osrm_url(coords: list) -> str:
    """Pick OSRM server based on centroid of coords. Falls back to UK server."""
    if not coords:
        return OSRM_URL
    clat = sum(c[0] for c in coords) / len(coords)
    clon = sum(c[1] for c in coords) / len(coords)
    for lat_min, lat_max, lon_min, lon_max, url in _OSRM_REGIONS:
        if lat_min <= clat <= lat_max and lon_min <= clon <= lon_max:
            return url
    return OSRM_URL  # default: UK server


def build_matrix(
    coords: List[Tuple[float, float]],
    api_key: str = "",
    mode: str = "driving",
) -> List[List[int]]:
    """
    Return n×n matrix of travel times in minutes.

    Priority:
      1. OSRM self-hosted (free, real road times, set OSRM_URL in .env)
      2. Google Maps Distance Matrix API (set GOOGLE_MAPS_API_KEY in .env)
      3. Haversine fallback (always available, ±15% accuracy)

    Parameters
    ----------
    coords  : list of (lat, lon) tuples — index 0 is the depot
    api_key : Google Maps API key override; falls back to GOOGLE_MAPS_API_KEY env var
    mode    : 'driving' | 'walking' | 'bicycling'
    """
    # Try OSRM first (auto-select UK vs US server by coordinate)
    osrm_url = _select_osrm_url(coords)
    if _osrm_available(osrm_url):
        try:
            return _osrm_matrix(coords, osrm_url)
        except Exception as exc:
            logger.warning("OSRM failed (%s) — trying next backend.", exc)

    # Try Google Maps
    key = api_key or GOOGLE_MAPS_API_KEY
    if key:
        try:
            return _gmaps_matrix(coords, key, mode)
        except Exception as exc:
            logger.warning("Google Maps API failed (%s) — falling back to haversine.", exc)

    return _haversine_matrix(coords)


# ── OSRM implementation ───────────────────────────────────────────────────────

def _osrm_available(url: str = OSRM_URL) -> bool:
    # OSRM has no /health endpoint; probe with a minimal route call
    try:
        r = requests.get(f"{url}/route/v1/driving/0,0;1,1", timeout=2)
        return r.status_code in (200, 400)   # 400 = server up, bad coords
    except Exception:
        return False


def _osrm_matrix(coords: List[Tuple[float, float]],
                 url: str = OSRM_URL) -> List[List[int]]:
    """
    Call OSRM Table API to get n×n duration matrix.
    OSRM uses lon,lat order (opposite to our lat,lon convention).
    Returns minutes (OSRM returns seconds).
    """
    n = len(coords)
    # OSRM Table API: /table/v1/driving/lon1,lat1;lon2,lat2;...
    coord_str  = ";".join(f"{lon},{lat}" for lat, lon in coords)
    table_url  = f"{url}/table/v1/driving/{coord_str}"
    params     = {"annotations": "duration"}

    resp = requests.get(table_url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if data.get("code") != "Ok":
        raise RuntimeError(f"OSRM Table API error: {data.get('code')} — {data.get('message','')}")

    durations = data["durations"]   # n×n seconds, None for unreachable
    matrix    = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            secs = durations[i][j]
            if secs is None:
                matrix[i][j] = _haversine_mins(coords[i], coords[j])
            else:
                matrix[i][j] = max(1, int(secs / 60))

    logger.info("Distance matrix built via OSRM (%d×%d).", n, n)
    return matrix


# ── Google Maps implementation ────────────────────────────────────────────────

def _gmaps_matrix(
    coords: List[Tuple[float, float]],
    api_key: str,
    mode: str,
) -> List[List[int]]:
    """
    Call Google Distance Matrix API in 10×10 batches (100 elements each).
    Fills any unroutable pairs with the haversine estimate.
    """
    n = len(coords)
    matrix = [[0] * n for _ in range(n)]

    for i_start in range(0, n, _BATCH_SIZE):
        i_end   = min(i_start + _BATCH_SIZE, n)
        origins = coords[i_start:i_end]

        for j_start in range(0, n, _BATCH_SIZE):
            j_end        = min(j_start + _BATCH_SIZE, n)
            destinations = coords[j_start:j_end]

            resp = _call_gmaps(origins, destinations, api_key, mode)

            for ri, row in enumerate(resp.get("rows", [])):
                for ci, element in enumerate(row.get("elements", [])):
                    gi, gj = i_start + ri, j_start + ci
                    if element.get("status") == "OK":
                        secs = element["duration"]["value"]
                        matrix[gi][gj] = max(1, secs // 60)
                    else:
                        matrix[gi][gj] = _haversine_mins(coords[gi], coords[gj])

    logger.info("Distance matrix built via Google Maps API (%d×%d).", n, n)
    return matrix


def _call_gmaps(
    origins: List[Tuple[float, float]],
    destinations: List[Tuple[float, float]],
    api_key: str,
    mode: str,
) -> dict:
    def fmt(pts: List[Tuple[float, float]]) -> str:
        return "|".join(f"{lat},{lon}" for lat, lon in pts)

    params = {
        "origins":        fmt(origins),
        "destinations":   fmt(destinations),
        "mode":           mode,
        "departure_time": "now",   # enables live-traffic durations
        "key":            api_key,
    }
    resp = requests.get(_GMAPS_URL, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    status = data.get("status")
    if status not in ("OK", "ZERO_RESULTS"):
        raise RuntimeError(
            f"Google Maps API error: {status} — {data.get('error_message', '')}"
        )
    return data


# ── Haversine fallback ────────────────────────────────────────────────────────

def _haversine_mins(a: Tuple[float, float], b: Tuple[float, float]) -> int:
    """Straight-line distance converted to minutes at _URBAN_SPEED_KMH."""
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = (math.sin(dlat / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
    km = 6371.0 * 2 * math.asin(math.sqrt(h))
    return max(1, round(km / _URBAN_SPEED_KMH * 60))


def _haversine_matrix(coords: List[Tuple[float, float]]) -> List[List[int]]:
    n = len(coords)
    matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i][j] = _haversine_mins(coords[i], coords[j])
    logger.info("Distance matrix built via haversine fallback (%d×%d).", n, n)
    return matrix


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample_coords = [
        (51.509865, -0.118092),   # Trafalgar Square (depot)
        (51.5238,   -0.1585),     # Baker Street
        (51.4975,   -0.1357),     # Victoria Station
        (51.5155,   -0.0922),     # St Paul's
    ]

    print("Haversine matrix (no API key required):")
    m = build_matrix(sample_coords, api_key="")
    for row in m:
        print(" ", row)

    if GOOGLE_MAPS_API_KEY:
        print("\nGoogle Maps matrix (real road times):")
        m2 = build_matrix(sample_coords)
        for row in m2:
            print(" ", row)
    else:
        print("\nGOOGLE_MAPS_API_KEY not set — set it in .env to test live API.")
