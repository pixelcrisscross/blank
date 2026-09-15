from __future__ import annotations

STATE_KEYS = {
    # Planning + location
    "plan": "orca_plan",
    "location": "orca_location",

    # Data blocks
    "ocean": "orca_ocean_data",
    "weather": "orca_weather_data",
    "geofence": "orca_geofence_data",
    "pfz": "orca_pfz_data",
    "coral": "orca_coral_data",
    "tides": "orca_tide_data",
    "biolum": "orca_biolum_data",
    "algal_bloom": "orca_algal_bloom_data",
    "imd": "orca_imd_data",                    # <-- NEW
    "risk": "orca_risk_assessment",

    # Reasoning blocks
    "ocean_reasoning": "orca_ocean_reasoning",
    "weather_reasoning": "orca_weather_reasoning",
    "fishery_reasoning": "orca_fishery_reasoning",
    "safety_reasoning": "orca_safety_reasoning",
    "tourism_reasoning": "orca_tourism_reasoning",

    # Review + final
    "peer_review": "orca_peer_review",
    "review": "orca_review",
    "final": "orca_final_response",
}


def key(name: str) -> str:
    return STATE_KEYS[name]