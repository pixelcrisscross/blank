"""
NOAA Coral Reef Watch bleaching-alert service with reef-presence check.

First checks whether the query point is near a known Indian reef region.
Only if a reef is present does it attempt to fetch NOAA CRW data.
Otherwise it returns NO_REEF_AT_LOCATION with the distance to the
nearest reef, so the agent can say something useful.
"""
from __future__ import annotations

import math

import httpx

# ── Indian reef regions (bounding boxes, approximate) ────────────────
# Source: SAC/ISRO reef maps, INCOIS reef atlases, published reef extents.
# For production, replace with true polygons from UNEP-WCMC or Allen Coral Atlas.
_INDIA_REEFS = [
    {
        "name": "Gulf of Kutch",
        "min_lat": 22.20, "max_lat": 22.80,
        "min_lon": 68.70, "max_lon": 70.60,
        "state": "Gujarat",
    },
    {
        "name": "Gulf of Mannar",
        "min_lat": 8.70, "max_lat": 9.50,
        "min_lon": 78.50, "max_lon": 79.40,
        "state": "Tamil Nadu",
    },
    {
        "name": "Lakshadweep",
        "min_lat": 8.00, "max_lat": 12.50,
        "min_lon": 71.50, "max_lon": 74.50,
        "state": "Lakshadweep",
    },
    {
        "name": "Andaman & Nicobar",
        "min_lat": 6.50, "max_lat": 14.00,
        "min_lon": 92.00, "max_lon": 94.50,
        "state": "Andaman & Nicobar Islands",
    },
    {
        "name": "Netrani Island",
        "min_lat": 14.00, "max_lat": 14.05,
        "min_lon": 74.30, "max_lon": 74.35,
        "state": "Karnataka",
    },
    {
        "name": "Malvan",
        "min_lat": 16.00, "max_lat": 16.10,
        "min_lon": 73.40, "max_lon": 73.55,
        "state": "Maharashtra",
    },
]

# ── ERDDAP mirrors (try in order) ────────────────────────────────────
_ERDDAP_MIRRORS = [
    "https://oceanwatch.pifsc.noaa.gov/erddap/griddap/noaacrwbaa7dDaily",
    "https://coastwatch.noaa.gov/erddap/griddap/noaacrwbaa7dDaily",
]

_LABELS = {
    0: "No Stress",
    1: "Bleaching Watch",
    2: "Bleaching Warning",
    3: "Bleaching Alert Level 1",
    4: "Bleaching Alert Level 2",
}

_INTERPRETATION = {
    0: "No significant heat stress on the reef.",
    1: "Heat stress is building — monitor conditions.",
    2: "Significant heat stress — bleaching possible.",
    3: "Severe heat stress — bleaching likely.",
    4: "Extreme heat stress — mass bleaching and mortality likely.",
}


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _nearest_reef(latitude: float, longitude: float) -> dict:
    """
    Return the nearest reef region and the distance to its bounding box.

    Distance is to the closest edge (0 if the point is inside).
    """
    best = None
    for reef in _INDIA_REEFS:
        # Clamp query point to bounding box to get nearest edge
        clamped_lat = max(reef["min_lat"], min(latitude, reef["max_lat"]))
        clamped_lon = max(reef["min_lon"], min(longitude, reef["max_lon"]))
        dist_km = _haversine_km(latitude, longitude, clamped_lat, clamped_lon)
        if best is None or dist_km < best["distance_km"]:
            best = {
                "name": reef["name"],
                "state": reef["state"],
                "distance_km": round(dist_km, 1),
                "inside": (
                    reef["min_lat"] <= latitude <= reef["max_lat"]
                    and reef["min_lon"] <= longitude <= reef["max_lon"]
                ),
            }
    return best


def _try_erddap(latitude: float, longitude: float) -> dict | None:
    """Try each ERDDAP mirror. Return None if all fail."""
    for base in _ERDDAP_MIRRORS:
        query = (
            f"{base}.json?"
            f"bleaching_alert_area"
            f"[(last)][({latitude}):1:({latitude})]"
            f"[({longitude}):1:({longitude})]"
        )
        try:
            r = httpx.get(query, timeout=45)
            r.raise_for_status()
            data = r.json()
            rows = data.get("table", {}).get("rows") or []
            raw = rows[0][-1] if rows else None
            if raw is None:
                continue
            return {"level": int(raw), "server": base.split("/")[2]}
        except Exception:
            continue
    return None


def _sst_heuristic(latitude: float, longitude: float) -> dict:
    """SST-based fallback when NOAA CRW is unreachable."""
    from tools.copernicus_service import get_copernicus_marine_snapshot
    snap = get_copernicus_marine_snapshot(latitude, longitude)
    obs = snap.get("observations", {}) or {}
    sst = (obs.get("temperature") or {}).get("value")

    if sst is None:
        return {
            "status": "NO_DATA",
            "source": "ORCA SST heuristic (NOAA CRW unreachable)",
            "note": "No SST value available for this location.",
        }

    if sst < 28.5:
        level = 0
    elif sst < 29.5:
        level = 1
    elif sst < 30.5:
        level = 2
    elif sst < 31.5:
        level = 3
    else:
        level = 4

    return {
        "status": "HEURISTIC",
        "source": "ORCA SST heuristic (NOAA CRW unreachable)",
        "sst_c": round(float(sst), 2),
        "bleaching_alert_level": level,
        "bleaching_alert": _LABELS.get(level, "Unknown"),
        "interpretation": _INTERPRETATION.get(level, "Unknown"),
        "note": (
            "NOAA Coral Reef Watch servers are unreachable from this network. "
            "This estimate is derived from SST alone and is NOT equivalent "
            "to the official NOAA Bleaching Alert Area."
        ),
    }


def get_coral_bleaching_alert(latitude: float, longitude: float) -> dict:
    """
    Reef-presence check, then NOAA CRW (or SST fallback).

    If no reef is near the query point, returns NO_REEF_AT_LOCATION with
    the distance to the nearest reef — no NOAA call is made.
    """
    nearest = _nearest_reef(latitude, longitude)

    # No reef within a generous radius → return early, no network call
    if not nearest["inside"] and nearest["distance_km"] > 50.0:
        return {
            "status": "NO_REEF_AT_LOCATION",
            "source": "ORCA reef-presence check",
            "query_point": {"latitude": latitude, "longitude": longitude},
            "nearest_reef": nearest["name"],
            "nearest_reef_state": nearest["state"],
            "distance_to_nearest_reef_km": nearest["distance_km"],
            "interpretation": (
                f"No coral reef is recorded within 50 km of this location. "
                f"The nearest reef region is {nearest['name']} "
                f"({nearest['state']}), approximately "
                f"{nearest['distance_km']} km away."
            ),
            "note": (
                "Coral bleaching alerts are only meaningful over reef areas. "
                "No NOAA Coral Reef Watch data was fetched for this point."
            ),
        }

    # Reef is present (or very close) — proceed with NOAA CRW
    result = _try_erddap(latitude, longitude)

    if result is not None:
        level = result["level"]
        return {
            "status": "OK",
            "source": f"NOAA Coral Reef Watch ({result['server']})",
            "query_point": {"latitude": latitude, "longitude": longitude},
            "reef_region": nearest["name"],
            "distance_to_reef_km": nearest["distance_km"],
            "bleaching_alert_level": level,
            "bleaching_alert": _LABELS.get(level, "Unknown"),
            "interpretation": _INTERPRETATION.get(level, "Unknown"),
            "note": (
                "Coral bleaching is driven by elevated SST. Prolonged heat "
                "stress damages reefs, which are critical nurseries for "
                "many commercial fish species."
            ),
        }

    # NOAA unreachable — SST fallback, still tagged with reef context
    fallback = _sst_heuristic(latitude, longitude)
    fallback["reef_region"] = nearest["name"]
    fallback["distance_to_reef_km"] = nearest["distance_km"]
    return fallback