"""
Algal-bloom risk classifier based on Copernicus chlorophyll-a.

Thresholds are tuned for coastal Indian waters. Not all blooms are harmful,
but elevated chlorophyll warrants caution for swimming and aquaculture.
"""
from __future__ import annotations

from tools.copernicus_service import get_copernicus_marine_snapshot


def _classify(chl: float) -> tuple[str, str]:
    if chl < 1:
        return "LOW", "Normal chlorophyll levels."
    if chl < 5:
        return "MODERATE", "Elevated chlorophyll — monitor for change."
    if chl < 10:
        return "ELEVATED", "Possible algal bloom forming."
    return "HIGH", "Algal bloom likely. Avoid swimming and shellfish harvest."


def get_algal_bloom_risk(latitude: float, longitude: float) -> dict:
    """Classify bloom risk from Copernicus chlorophyll-a concentration."""
    snap = get_copernicus_marine_snapshot(latitude, longitude)
    obs = snap.get("observations", {}) or {}
    chl_block = obs.get("chlorophyll") or {}
    value = chl_block.get("value")

    if value is None:
        return {
            "status": "NO_DATA",
            "source": "Copernicus Marine chlorophyll-a",
            "query_point": {"latitude": latitude, "longitude": longitude},
            "note": "No chlorophyll-a value available for this location.",
        }

    level, interpretation = _classify(float(value))

    return {
        "status": "OK",
        "source": "Copernicus Marine (GlobColour L4)",
        "query_point": {"latitude": latitude, "longitude": longitude},
        "chlorophyll_a_mg_m3": value,
        "risk_level": level,
        "interpretation": interpretation,
        "observation_time": chl_block.get("observation_time"),
        "note": (
            "Elevated chlorophyll-a in coastal waters is a strong indicator "
            "of algal blooms. Not all blooms produce toxins, but some do."
        ),
    }