"""
Coastal-proximity check.

Thin wrapper over tools.nearest_coast for backwards compatibility.
"""
from __future__ import annotations

from tools.nearest_coast import nearest_coast_info


def is_coastal(latitude: float, longitude: float,
               threshold_km: float = 30.0) -> dict:
    """
    Classify whether a point is near the coast.

    Returns:
        coastal        : bool
        distance_km    : distance to nearest ocean (0 if in water)
        threshold_km   : the threshold used
        method         : "ocean" if the point is in water, "land" otherwise
    """
    info = nearest_coast_info(latitude, longitude, max_distance_km=threshold_km)
    return {
        "coastal": info["is_coastal"] or (
            info["found"] and info["distance_km"] <= threshold_km
        ),
        "distance_km": info["distance_km"],
        "threshold_km": threshold_km,
        "method": "ocean" if info["is_coastal"] else "land",
    }