"""
Find the nearest ocean point to any location.

Uses global-land-mask (already a dependency) with an expanding-ring
spiral search. Returns the actual lat/lon of the nearest ocean point,
so downstream marine tools can query it directly without any hardcoded
location tables.
"""
from __future__ import annotations

import math

from global_land_mask import globe


# Marine tools are allowed to use a nearest-coast point up to this
# distance from the original query point. Beyond this, the query is
# treated as deep-inland and marine domains are skipped.
DEFAULT_MAX_MARINE_DISTANCE_KM = 200.0


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def nearest_ocean_point(
    latitude: float,
    longitude: float,
    max_distance_km: float = 500.0,
) -> tuple[float, float, float] | None:
    """
    Return (lat, lon, distance_km) of the nearest ocean point.

    Returns None if no ocean is found within max_distance_km.

    Strategy: expanding concentric rings around the query point.
    """
    if globe.is_ocean(latitude, longitude):
        return (latitude, longitude, 0.0)

    cos_lat = max(0.1, math.cos(math.radians(latitude)))
    max_radius_deg = max_distance_km / 111.0

    radii_deg = [
        0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75,
        0.9, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0,
    ]

    for radius in radii_deg:
        if radius > max_radius_deg:
            break
        samples = max(24, int(radius * 120))
        best = None
        for i in range(samples):
            angle = 2 * math.pi * i / samples
            dlat = radius * math.cos(angle)
            dlon = radius * math.sin(angle) / cos_lat
            test_lat = latitude + dlat
            test_lon = longitude + dlon

            if not (-90 <= test_lat <= 90):
                continue
            if test_lon > 180:
                test_lon -= 360
            elif test_lon < -180:
                test_lon += 360

            if globe.is_ocean(test_lat, test_lon):
                d = _haversine_km(latitude, longitude, test_lat, test_lon)
                if best is None or d < best[2]:
                    best = (test_lat, test_lon, d)

        if best is not None:
            return best

    return None


def nearest_coast_info(
    latitude: float,
    longitude: float,
    max_distance_km: float = DEFAULT_MAX_MARINE_DISTANCE_KM,
) -> dict:
    """
    Return a structured description of the nearest coast.

    Fields:
        is_coastal        : bool, True if the query point itself is ocean
        found             : bool, True if a coast was found within range
        coastal_point     : dict {latitude, longitude} or None
        distance_km       : float or None
        within_range      : bool, True if distance ≤ max_distance_km
        note              : human-readable description
    """
    if globe.is_ocean(latitude, longitude):
        return {
            "is_coastal": True,
            "found": True,
            "coastal_point": {
                "latitude": round(latitude, 4),
                "longitude": round(longitude, 4),
            },
            "distance_km": 0.0,
            "within_range": True,
            "note": "Query point is on the ocean.",
        }

    search_radius = max(max_distance_km * 2.5, 500.0)
    result = nearest_ocean_point(latitude, longitude, search_radius)

    if result is None:
        return {
            "is_coastal": False,
            "found": False,
            "coastal_point": None,
            "distance_km": None,
            "within_range": False,
            "note": (
                f"No ocean found within {search_radius:.0f} km. "
                f"Location is deep inland."
            ),
        }

    c_lat, c_lon, dist = result
    within = dist <= max_distance_km

    if within:
        note = (
            f"Query point is {dist:.1f} km inland. "
            f"Nearest coast is at ({c_lat:.4f}, {c_lon:.4f})."
        )
    else:
        note = (
            f"Query point is {dist:.1f} km inland — beyond the "
            f"{max_distance_km:.0f} km marine-data range. "
            f"Marine data will be skipped."
        )

    return {
        "is_coastal": False,
        "found": True,
        "coastal_point": {
            "latitude": round(c_lat, 4),
            "longitude": round(c_lon, 4),
        },
        "distance_km": round(dist, 1),
        "within_range": within,
        "note": note,
    }