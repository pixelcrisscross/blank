"""
ORCA Geofence Service.

Checks whether a given coordinate is inside or near a set of maritime
boundary zones relevant to Indian coastal operations.

Zones encoded as lightweight circle geometries (centre + radius) —
no external GIS file required. Proximity warning triggers when within
10 km of a zone boundary.

Zone types
----------
IMBL          : India–Sri Lanka International Maritime Boundary Line
                sensitive zone (Palk Strait / Gulf of Mannar area)
MPA           : Marine Protected Area / Marine National Park
EEZ_BUFFER    : Approach buffer zone near Exclusive Economic Zone limit
ECOLOGICALLY_SENSITIVE : Areas with high biodiversity / coral cover
RESTRICTED    : Areas closed to fishing or navigation by Indian authorities

Competition brief requirement:
  "Providing geofencing-based notifications when approaching international
   maritime boundaries, restricted waters, marine protected areas,
   ecologically sensitive zones, or other predefined operational boundaries."

Attribution: Zone centres derived from publicly available Indian
government GIS data and MarineRegions.org (VLIZ, CC-BY). Radii are
approximate and intended for demonstration; use authoritative survey
data for operational use.
"""
from __future__ import annotations

from math import atan2, cos, radians, sin, sqrt
from typing import Any


# ─────────────────────────────────────────────────────────────────────
# ZONE DEFINITIONS
# Each zone: name, latitude, longitude, radius_km, type, authority
# ─────────────────────────────────────────────────────────────────────

MARITIME_ZONES: list[dict[str, Any]] = [
    # ── India–Sri Lanka IMBL / Palk Strait sensitive zone ───────────
    {
        "name": "Palk Strait IMBL Sensitive Zone",
        "latitude": 9.55,
        "longitude": 80.0,
        "radius_km": 35.0,
        "type": "IMBL",
        "authority": "India–Sri Lanka International Maritime Boundary Line",
        "description": (
            "Sensitive zone near the India–Sri Lanka International Maritime "
            "Boundary Line in the Palk Strait. Indian fishing vessels must "
            "not cross this boundary."
        ),
    },
    # ── Gulf of Mannar Marine National Park ─────────────────────────
    {
        "name": "Gulf of Mannar Marine National Park",
        "latitude": 8.97,
        "longitude": 78.15,
        "radius_km": 28.0,
        "type": "MPA",
        "authority": "Ministry of Environment, Forest and Climate Change (MoEFCC)",
        "description": (
            "Designated Marine National Park with 21 islands and coral "
            "reef ecosystems. Fishing and anchoring are restricted within "
            "the core zone."
        ),
    },
    # ── Lakshadweep EEZ approach / IMBL buffer ──────────────────────
    {
        "name": "Lakshadweep EEZ Approach Zone",
        "latitude": 10.56,
        "longitude": 72.64,
        "radius_km": 40.0,
        "type": "EEZ_BUFFER",
        "authority": "Indian Navy / Coast Guard",
        "description": (
            "Approach buffer zone near the Lakshadweep Islands. Vessels "
            "should verify their position relative to the Indian EEZ limit "
            "and Maldives boundary."
        ),
    },
    # ── Andaman EEZ approach / sensitive zone ───────────────────────
    {
        "name": "Andaman & Nicobar EEZ Approach Zone",
        "latitude": 10.66,
        "longitude": 92.72,
        "radius_km": 50.0,
        "type": "EEZ_BUFFER",
        "authority": "Indian Navy",
        "description": (
            "Approach buffer zone near the Andaman & Nicobar Islands. "
            "Vessels should remain within the Indian EEZ and avoid the "
            "Myanmar maritime boundary."
        ),
    },
    # ── Sundarbans Biosphere Reserve (Bay of Bengal) ─────────────────
    {
        "name": "Sundarbans Marine Biosphere Reserve",
        "latitude": 21.9,
        "longitude": 89.1,
        "radius_km": 30.0,
        "type": "ECOLOGICALLY_SENSITIVE",
        "authority": "UNESCO / MoEFCC",
        "description": (
            "UNESCO World Heritage Site. Mangrove-rich marine biosphere. "
            "Industrial fishing and harmful gear are prohibited in the "
            "core zone."
        ),
    },
    # ── Malvan Marine Sanctuary (Maharashtra) ───────────────────────
    {
        "name": "Malvan Marine Sanctuary",
        "latitude": 16.06,
        "longitude": 73.46,
        "radius_km": 12.0,
        "type": "MPA",
        "authority": "Maharashtra Forest Department",
        "description": (
            "Notified Marine Sanctuary with coral reefs, seagrass, and "
            "diverse reef fish. Fishing and anchoring restricted."
        ),
    },
    # ── Vembanad–Kol Ramsar Wetland (Kerala coast) ──────────────────
    {
        "name": "Vembanad–Kol Wetland (Ramsar) Coastal Zone",
        "latitude": 9.6,
        "longitude": 76.35,
        "radius_km": 15.0,
        "type": "ECOLOGICALLY_SENSITIVE",
        "authority": "Ramsar Convention / Kerala Government",
        "description": (
            "Ramsar-designated wetland and backwater ecosystem. "
            "Industrial trawling is restricted in the nearshore zone."
        ),
    },
    # ── Chilika Lake Ramsar Wetland (Odisha) ─────────────────────────
    {
        "name": "Chilika Lake Ramsar Wetland",
        "latitude": 19.75,
        "longitude": 85.32,
        "radius_km": 20.0,
        "type": "ECOLOGICALLY_SENSITIVE",
        "authority": "Ramsar Convention / Odisha Government",
        "description": (
            "Asia's largest brackish water lagoon. Ramsar site supporting "
            "irrawaddy dolphins and migratory birds. Mechanical trawling "
            "is prohibited."
        ),
    },
]

# Proximity warning triggers this many km from a zone boundary
_PROXIMITY_KM = 10.0


# ─────────────────────────────────────────────────────────────────────
# GEOMETRY
# ─────────────────────────────────────────────────────────────────────

def _distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    earth_radius = 6371.0
    p1, p2 = radians(lat1), radians(lat2)
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(p1) * cos(p2) * sin(dlon / 2) ** 2
    return 2 * earth_radius * atan2(sqrt(a), sqrt(1 - a))


# ─────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────

def check_geofence(latitude: float, longitude: float) -> dict[str, Any]:
    """
    Check a coordinate against all defined maritime zones.

    Returns
    -------
    dict with:
        status                : "OK"
        inside_restricted_zone: bool — inside any zone
        matches               : list of zones the point is inside
        proximity_warning     : bool — within 10 km of a zone boundary
        proximity_warnings    : list of zones within 10 km of boundary
        source                : attribution string
        data_quality          : "REPRESENTATIVE" (not survey-grade)
        warning               : disclaimer
    """
    matches: list[dict[str, Any]] = []
    proximity_warnings: list[dict[str, Any]] = []

    for zone in MARITIME_ZONES:
        dist = _distance_km(
            latitude, longitude,
            zone["latitude"], zone["longitude"],
        )

        if dist <= zone["radius_km"]:
            matches.append({
                "name": zone["name"],
                "type": zone["type"],
                "authority": zone["authority"],
                "distance_to_centre_km": round(dist, 2),
                "description": zone["description"],
            })
        elif dist <= zone["radius_km"] + _PROXIMITY_KM:
            # Within proximity buffer but not inside
            margin = round(dist - zone["radius_km"], 2)
            proximity_warnings.append({
                "name": zone["name"],
                "type": zone["type"],
                "authority": zone["authority"],
                "distance_to_boundary_km": margin,
                "description": zone["description"],
            })

    return {
        "status": "OK",
        "inside_restricted_zone": bool(matches),
        "matches": matches,
        "proximity_warning": bool(proximity_warnings),
        "proximity_warnings": proximity_warnings,
        "source": (
            "ORCA Geofence Layer — IMBL, MPA, EEZ buffer, and "
            "ecologically sensitive zones for Indian coastal waters."
        ),
        "data_quality": "REPRESENTATIVE",
        "warning": (
            "Zone boundaries are representative and based on publicly "
            "available data. Always cross-check with official Indian "
            "Coast Guard and Survey of India charts before navigation."
        ),
    }