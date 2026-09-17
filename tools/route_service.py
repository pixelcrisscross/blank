"""
Route Safety Service for ORCA.

Samples waypoints along a great-circle route at fixed intervals and
evaluates weather and geofence risk at each point.

Competition brief requirement:
  "What is the safest route for a fishing vessel considering weather
   and sea-state conditions?"
  "Assisting with route optimization, safe navigation, and operational
   planning based on prevailing and forecast marine conditions."

No external GIS libraries required — uses pure spherical geometry.
"""
from __future__ import annotations

import math
from typing import Any


# ─────────────────────────────────────────────────────────────────────
# GREAT-CIRCLE WAYPOINT SAMPLER
# ─────────────────────────────────────────────────────────────────────

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance between two points in km."""
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2)
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _intermediate_point(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
    fraction: float,
) -> tuple[float, float]:
    """
    Spherical-linear interpolation between two points on a great circle.
    fraction 0.0 = start, 1.0 = end.
    """
    lat1r, lon1r = math.radians(lat1), math.radians(lon1)
    lat2r, lon2r = math.radians(lat2), math.radians(lon2)

    d = 2 * math.asin(math.sqrt(
        math.sin((lat2r - lat1r) / 2) ** 2
        + math.cos(lat1r) * math.cos(lat2r)
        * math.sin((lon2r - lon1r) / 2) ** 2
    ))

    if d < 1e-10:
        # Points are essentially identical
        return (lat1, lon1)

    sin_d = math.sin(d)
    A = math.sin((1 - fraction) * d) / sin_d
    B = math.sin(fraction * d) / sin_d

    x = A * math.cos(lat1r) * math.cos(lon1r) + B * math.cos(lat2r) * math.cos(lon2r)
    y = A * math.cos(lat1r) * math.sin(lon1r) + B * math.cos(lat2r) * math.sin(lon2r)
    z = A * math.sin(lat1r) + B * math.sin(lat2r)

    lat_out = math.degrees(math.atan2(z, math.sqrt(x * x + y * y)))
    lon_out = math.degrees(math.atan2(y, x))
    return (round(lat_out, 5), round(lon_out, 5))


def sample_route_waypoints(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    step_km: float = 25.0,
    max_points: int = 10,
) -> list[dict[str, Any]]:
    """
    Sample up to `max_points` waypoints at `step_km` intervals along
    the great-circle route from (start_lat, start_lon) to (end_lat, end_lon).

    Always includes start and end. Intermediate points are spaced at
    step_km intervals. If the total distance < step_km, only start and
    end are returned.

    Returns list of dicts: {
        label: "START" | "WP-1" | ... | "END",
        latitude: float,
        longitude: float,
        distance_from_start_km: float,
    }
    """
    total_km = _haversine_km(start_lat, start_lon, end_lat, end_lon)

    waypoints: list[dict[str, Any]] = []
    waypoints.append({
        "label": "START",
        "latitude": round(start_lat, 5),
        "longitude": round(start_lon, 5),
        "distance_from_start_km": 0.0,
    })

    if total_km <= step_km:
        # Route shorter than one step — just start + end
        waypoints.append({
            "label": "END",
            "latitude": round(end_lat, 5),
            "longitude": round(end_lon, 5),
            "distance_from_start_km": round(total_km, 1),
        })
        return waypoints

    # Number of intermediate steps (capped at max_points - 2 to leave
    # room for start and end)
    n_intermediate = min(
        max(1, int(total_km / step_km) - 1),
        max_points - 2,
    )
    actual_step = total_km / (n_intermediate + 1)

    for i in range(1, n_intermediate + 1):
        fraction = i * actual_step / total_km
        ilat, ilon = _intermediate_point(
            start_lat, start_lon, end_lat, end_lon, fraction
        )
        waypoints.append({
            "label": f"WP-{i}",
            "latitude": ilat,
            "longitude": ilon,
            "distance_from_start_km": round(i * actual_step, 1),
        })

    waypoints.append({
        "label": "END",
        "latitude": round(end_lat, 5),
        "longitude": round(end_lon, 5),
        "distance_from_start_km": round(total_km, 1),
    })
    return waypoints


# ─────────────────────────────────────────────────────────────────────
# RISK COLOUR HELPERS
# ─────────────────────────────────────────────────────────────────────

_WEATHER_CODE_RISK: dict[range, str] = {}

def _weather_risk_label(
    wind_speed: float | None,
    wind_gust: float | None,
    weather_code: float | None,
    precipitation: float | None,
) -> str:
    """
    Simple risk label for a route waypoint.
    Returns: "LOW" | "MODERATE" | "HIGH" | "UNKNOWN"
    """
    score = 0
    if wind_speed is not None:
        if wind_speed >= 45:
            score += 3
        elif wind_speed >= 30:
            score += 2
        elif wind_speed >= 20:
            score += 1
    if wind_gust is not None and wind_gust >= 45:
        score += 2
    if weather_code is not None and weather_code >= 80:
        score += 2
    if precipitation is not None and precipitation >= 10:
        score += 1

    if score == 0:
        return "LOW"
    if score <= 2:
        return "MODERATE"
    return "HIGH"


def summarise_route_risk(
    waypoint_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Aggregate waypoint-level risk into an overall route risk assessment.

    Parameters
    ----------
    waypoint_results : list of dicts, each containing:
        label, latitude, longitude, distance_from_start_km,
        risk_level, geofence_inside, weather (optional dict)

    Returns
    -------
    dict with overall_risk, hazard_segments, safe_window, note
    """
    if not waypoint_results:
        return {
            "overall_risk": "UNKNOWN",
            "hazard_segments": [],
            "note": "No waypoints evaluated.",
        }

    risk_order = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "BLOCKED": 3, "UNKNOWN": -1}
    max_risk = max(
        (r.get("risk_level", "UNKNOWN") for r in waypoint_results),
        key=lambda r: risk_order.get(r, -1),
        default="UNKNOWN",
    )

    hazard_segments = [
        {
            "waypoint": r["label"],
            "latitude": r["latitude"],
            "longitude": r["longitude"],
            "distance_km": r.get("distance_from_start_km"),
            "risk_level": r.get("risk_level"),
            "risk_reason": r.get("risk_reason"),
            "geofence_alert": r.get("geofence_inside") or r.get("geofence_proximity"),
        }
        for r in waypoint_results
        if risk_order.get(r.get("risk_level", "LOW"), 0) >= 1
        or r.get("geofence_inside")
        or r.get("geofence_proximity")
    ]

    return {
        "overall_risk": max_risk,
        "total_waypoints": len(waypoint_results),
        "hazard_segments": hazard_segments,
        "note": (
            "Route risk assessed using Open-Meteo weather and ORCA "
            "geofence zones at each waypoint. Decision support only — "
            "always verify with official marine advisories."
        ),
    }
