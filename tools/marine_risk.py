"""
Deterministic marine risk engine.

Three-layer design:
    Layer A — Hard blockers (official alerts → BLOCKED)
    Layer B — Environmental score (waves / wind / gust / precip / code)
    Layer C — Decision: BLOCKED if any blocker, else LOW → VERY HIGH

Blocker guards:
- Only local alerts can block. Alerts flagged india_wide are ignored.
- Only state-matched alerts count. A state_hint is required.
"""
from __future__ import annotations


def _num(value):
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _is_local_alert(block: dict | None) -> bool:
    """
    Return True only if a hint-based tool returned a state-matched,
    non-India-wide alert.
    """
    if not isinstance(block, dict):
        return False
    if block.get("status") != "OK":
        return False
    if block.get("india_wide"):
        return False
    if block.get("max_severity") in (None, "NONE", "UNKNOWN"):
        return False
    return True


def calculate_marine_risk(
    ocean: dict,
    weather: dict,
    geofence: dict,
    hwassa: dict | None = None,
    cyclone: dict | None = None,
    tsunami: dict | None = None,
    currents: dict | None = None,
) -> dict:
    blockers: list[str] = []
    score = 0
    reasons: list[str] = []

    # ── LAYER A — HARD BLOCKERS ──────────────────────────────────────
    # HWA / SSA — only local Orange alerts block.
    if hwassa and _is_local_alert(hwassa):
        if hwassa.get("max_severity") == "ALERT":
            blockers.append(
                f"INCOIS {hwassa.get('alert_type', 'alert')} for "
                f"{hwassa.get('district')}, {hwassa.get('state')}: "
                f"{(hwassa.get('message') or '')[:200]}"
            )

    # Cyclone — always local since the API returns active systems.
    if cyclone:
        alerts = cyclone.get("active_alerts")
        if isinstance(alerts, list) and alerts:
            blockers.append(
                f"Active cyclone alert(s): {len(alerts)}. "
                f"Check IMD for track and intensity."
            )

    # Tsunami — global, applies to the whole Indian coast.
    if tsunami and tsunami.get("threat_to_india"):
        threat = tsunami.get("threat_events") or []
        if threat:
            first = threat[0]
            blockers.append(
                f"Tsunami threat to India declared by INCOIS ITEWS: "
                f"M{first.get('magnitude')} at {first.get('region')} — "
                f"{first.get('evaluation')}"
            )
        else:
            blockers.append("Tsunami threat to India declared by INCOIS ITEWS.")

    # Geofence — only if the point actually intersects a zone.
    if isinstance(geofence, dict) and geofence.get("inside_restricted_zone"):
        blockers.append("Location intersects a configured restricted zone.")

    # ── LAYER B — ENVIRONMENTAL SCORE ────────────────────────────────
    observations = ocean.get("observations", {}) if isinstance(ocean, dict) else {}
    waves = observations.get("waves", {}) or {}
    weather_current = weather.get("current", {}) if isinstance(weather, dict) else {}

    wave_height = _num(waves.get("significant_wave_height_m"))
    wind_speed = _num(weather_current.get("wind_speed_10m"))
    wind_gust = _num(weather_current.get("wind_gusts_10m"))
    precipitation = _num(weather_current.get("precipitation"))
    weather_code = _num(weather_current.get("weather_code"))

    if wave_height is not None:
        if wave_height >= 2.5:
            score += 40; reasons.append("Very high significant wave height.")
        elif wave_height >= 1.5:
            score += 25; reasons.append("Elevated significant wave height.")
        elif wave_height >= 1.0:
            score += 10; reasons.append("Moderate wave conditions.")

    if wind_speed is not None:
        if wind_speed >= 45:
            score += 35; reasons.append("Strong wind conditions.")
        elif wind_speed >= 30:
            score += 20; reasons.append("Elevated wind conditions.")
        elif wind_speed >= 20:
            score += 8; reasons.append("Moderate wind conditions.")

    if wind_gust is not None and wind_gust >= 45:
        score += 15; reasons.append("Strong wind gust signal.")

    if weather_code is not None and weather_code >= 80:
        score += 20; reasons.append("Heavy-weather precipitation code detected.")

    if precipitation is not None and precipitation >= 10:
        score += 10; reasons.append("Heavy precipitation signal.")

    # Currents — only local advisories contribute.
    if _is_local_alert(currents):
        if currents.get("max_severity") == "ALERT":
            score += 15
            reasons.append(
                f"INCOIS Ocean Current Alert for "
                f"{currents.get('district')} — caution advised."
            )
        elif currents.get("max_severity") == "WATCH":
            score += 5
            reasons.append(
                f"INCOIS Ocean Current Watch for "
                f"{currents.get('district')}."
            )

    evaluated = {
        "wave_height_m": wave_height,
        "wind_speed": wind_speed,
        "wind_gust": wind_gust,
        "weather_code": weather_code,
        "precipitation_mm": precipitation,
    }

    # ── LAYER C — DECISION ──────────────────────────────────────────
    if blockers:
        return {
            "risk_level": "BLOCKED",
            "risk_score": 100,
            "blockers": blockers,
            "reasons": reasons,
            "evaluated_parameters": evaluated,
            "decision_support_only": True,
            "warning": (
                "One or more official hazards are active for this "
                "location. Do not proceed without checking authorities."
            ),
        }

    score = min(score, 100)
    if score < 25:
        level = "LOW"
    elif score < 55:
        level = "MODERATE"
    elif score < 80:
        level = "HIGH"
    else:
        level = "VERY HIGH"

    return {
        "risk_level": level,
        "risk_score": score,
        "blockers": [],
        "reasons": reasons,
        "evaluated_parameters": evaluated,
        "decision_support_only": True,
        "warning": (
            "Heuristic decision support; official marine warnings take "
            "precedence."
        ),
    }