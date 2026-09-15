"""
Coastal-proximity check using a global land/ocean raster.

Uses the `global-land-mask` package — a pre-computed 1 km global mask
derived from NOAA GLOBE elevation data. No hardcoded coordinates, no
network calls at runtime.

For a land point, we search outward in rings to find the nearest ocean
cell, then compute the geodesic distance to it.

Install:
    pip install global-land-mask
"""
from __future__ import annotations

import math

from global_land_mask import globe


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _nearest_ocean_km(latitude: float, longitude: float) -> float | None:
    """
    Search outward in expanding rings for the nearest ocean cell.
    Returns distance in km, or None if no ocean found within ~220 km.
    """
    # Ring radii in degrees: 0.1° ≈ 11 km, 2.0° ≈ 220 km
    radii = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
    for radius in radii:
        samples = max(8, int(radius * 60))
        for i in range(samples):
            angle = 2 * math.pi * i / samples
            dlat = radius * math.cos(angle)
            dlon = (
                radius * math.sin(angle)
                / max(0.1, math.cos(math.radians(latitude)))
            )
            if globe.is_ocean(latitude + dlat, longitude + dlon):
                return _haversine_km(
                    latitude, longitude,
                    latitude + dlat, longitude + dlon,
                )
    return None


def is_coastal(latitude: float, longitude: float,
               threshold_km: float = 30.0) -> dict:
    """
    Classify whether a point is near the coast.

    Returns:
        coastal        : bool
        distance_km    : approximate distance to nearest ocean (0 if in water)
        threshold_km   : the threshold used
        method         : "ocean" if the point is in water, "land" otherwise
    """
    if globe.is_ocean(latitude, longitude):
        return {
            "coastal": True,
            "distance_km": 0.0,
            "threshold_km": threshold_km,
            "method": "ocean",
        }

    d = _nearest_ocean_km(latitude, longitude)
    if d is None:
        return {
            "coastal": False,
            "distance_km": None,
            "threshold_km": threshold_km,
            "method": "land",
        }

    return {
        "coastal": d <= threshold_km,
        "distance_km": round(d, 1),
        "threshold_km": threshold_km,
        "method": "land",
    }